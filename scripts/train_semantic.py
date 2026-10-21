import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from memora.dialogue import (
    CONTROL_TOKENS,
    VOCABULARY,
    atomic_json,
    expand_vocabulary,
    load_semantic_sequence,
)
from memora.paths import MODEL_DIR, corpus_dir
from memora.semantic import MemoraConfig, MemoraLanguageModel


def groups(rows, batch, epoch):
    order = sorted(range(len(rows)), key=lambda i: sum(t["frames"] + 2 for t in rows[i]["turns"]))
    result = [order[i : i + batch] for i in range(0, len(order), batch)]
    random.Random(42 + epoch).shuffle(result)
    return result


def batches(rows, indices, context, device):
    sequences = [load_semantic_sequence(rows[i]) for i in indices]
    for start in range(0, max(len(x[0]) for x in sequences) - 1, context):
        length = min(context, max(len(x[0]) for x in sequences) - 1 - start)
        x = torch.zeros(len(indices), length, dtype=torch.long, device=device)
        y = torch.zeros_like(x)
        mask = torch.zeros_like(x, dtype=torch.bool)
        for i, (tokens, supervised) in enumerate(sequences):
            count = min(length, len(tokens) - 1 - start)
            if count > 0:
                x[i, :count] = torch.as_tensor(tokens[start : start + count], device=device)
                y[i, :count] = torch.as_tensor(tokens[start + 1 : start + count + 1], device=device)
                mask[i, :count] = torch.as_tensor(
                    supervised[start + 1 : start + count + 1], device=device
                )
        yield x, y, mask


@torch.inference_mode()
def evaluate(model, rows, batch, context, device, maximum=100000):
    model.eval()
    loss = correct = top5 = total = end_total = end_correct = 0
    audio_loss = audio_count = early_end = 0
    order = list(range(len(rows)))
    random.Random(987).shuffle(order)
    for offset in range(0, len(order), batch):
        state = None
        for x, y, mask in batches(rows, order[offset : offset + batch], context, device):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(x, state=state)
            state = out["state"]
            logits = out["logits"].float()
            ce = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="none").view_as(mask)
            n = int(mask.sum())
            loss += float((ce * mask).sum())
            correct += int(((logits.argmax(-1) == y) & mask).sum())
            top5 += int(((logits.topk(5, -1).indices == y[..., None]).any(-1) & mask).sum())
            total += n
            end = mask & (y == CONTROL_TOKENS["<end_turn>"])
            end_total += int(end.sum())
            end_correct += int(((logits.argmax(-1) == y) & end).sum())
            audio = mask & (y < 2048)
            audio_loss += float((ce * audio).sum())
            audio_count += int(audio.sum())
            early_end += int(((logits.argmax(-1) == CONTROL_TOKENS["<end_turn>"]) & audio).sum())
        if total >= maximum:
            break
    if not total:
        raise ValueError("No supervised validation targets")
    model.train()
    ce = loss / total
    return {
        "cross_entropy": ce,
        "perplexity": math.exp(ce),
        "top1": correct / total,
        "top5": top5 / total,
        "assistant_targets": total,
        "audio_perplexity": math.exp(audio_loss / max(audio_count, 1)),
        "premature_end_turn_rate": early_end / max(audio_count, 1),
        "end_turn_targets": end_total,
        "end_turn_top1": end_correct / max(1, end_total),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=corpus_dir())
    p.add_argument("--initial", type=Path)
    p.add_argument("--output", type=Path, default=MODEL_DIR / "semantic")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--balanced-dialogues", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--max-updates", type=int)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    device = torch.device(a.device)
    torch.manual_seed(42)
    train = json.loads((a.data / "train.json").read_text())
    valid = json.loads((a.data / "valid.json").read_text())
    if a.limit:
        train = train[: a.limit]
        valid = valid[: a.limit]
    if {r["id"] for r in train} & {r["id"] for r in valid}:
        raise ValueError("Conversation leakage")
    a.output.mkdir(parents=True, exist_ok=True)
    checkpoint = a.output / "checkpoint.pt"
    if checkpoint.exists() and not a.resume:
        raise ValueError("Checkpoint exists; explicitly use --resume")
    source = checkpoint if a.resume and checkpoint.exists() else a.initial
    pkg = None
    if source is not None:
        pkg = torch.load(source, map_location="cpu", weights_only=True)
        model = expand_vocabulary(MemoraLanguageModel.from_package(pkg)).to(device)
    else:
        model = MemoraLanguageModel(MemoraConfig(vocabulary=VOCABULARY, context=a.context)).to(
            device
        )
    if device.type == "cuda":
        model = model.to(torch.bfloat16)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=a.learning_rate, betas=(0.9, 0.95), weight_decay=0.1
    )
    resumed = source == checkpoint and pkg is not None
    if resumed:
        optimizer.load_state_dict(pkg["optimizer"])
    updates = pkg.get("updates", 0) if resumed else 0
    first_epoch = pkg.get("next_epoch", 1) if resumed else 1
    first_group = pkg.get("next_group", 0) if resumed else 0
    best = pkg.get("best_cross_entropy", math.inf) if resumed else math.inf
    started = time.monotonic()
    recent = []
    supervised_tokens = 0
    input_tokens = 0

    def log(kind, **kw):
        row = {"kind": kind, "updates": updates, "seconds": time.monotonic() - started, **kw}
        with (a.output / "metrics.jsonl").open("a") as out:
            out.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    initial = evaluate(model, valid, a.batch_size, a.context, device)
    log(
        "initial",
        parameters=sum(p.numel() for p in model.parameters()),
        source=str(source),
        train_conversations=len(train),
        validation_conversations=len(valid),
        control_tokens=CONTROL_TOKENS,
        **initial,
    )

    def save(epoch, group, score=None):
        package = model.package(
            updates=updates,
            next_epoch=epoch,
            next_group=group,
            optimizer=optimizer.state_dict(),
            best_cross_entropy=best,
            control_tokens=CONTROL_TOKENS,
            validation=score,
        )
        tmp = a.output / "checkpoint.partial.pt"
        torch.save(package, tmp)
        tmp.replace(checkpoint)
        return package

    for epoch in range(first_epoch, a.epochs + 1):
        grouped = groups(train, a.batch_size, epoch)
        for gi, indices in enumerate(grouped):
            if epoch == first_epoch and gi < first_group:
                continue
            state = None
            window_batches = list(batches(train, indices, a.context, device))
            trainable = sum(bool(mask.any()) for _, _, mask in window_batches)
            if a.balanced_dialogues:
                optimizer.zero_grad(set_to_none=True)
            group_had_loss = False
            for x, y, mask in window_batches:
                if not a.balanced_dialogues:
                    optimizer.zero_grad(set_to_none=True)
                count = int(mask.sum())
                input_tokens += x.numel()
                with torch.set_grad_enabled(count > 0):
                    with torch.autocast(
                        device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
                    ):
                        out = model(x, state=state, retain=True)
                        if count:
                            ce = F.cross_entropy(
                                out["logits"].float().flatten(0, 1), y.flatten(), reduction="none"
                            ).view_as(mask)
                            prediction = (ce * mask).sum() / count
                            loss = prediction + 0.01 * out["memory_loss"]
                    state = out["state"].detached()
                    if count:
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite dialogue loss")
                        (loss / trainable if a.balanced_dialogues else loss).backward()
                        group_had_loss = True
                        supervised_tokens += count
                        recent.append(float(prediction.detach()))
                        if len(recent) > 100:
                            recent.pop(0)
                        if not a.balanced_dialogues:
                            gradient = float(
                                torch.nn.utils.clip_grad_norm_(
                                    model.parameters(), 1, error_if_nonfinite=True
                                )
                            )
                            optimizer.step()
                            updates += 1
                            if updates == 1 or updates % 100 == 0:
                                log(
                                    "train",
                                    epoch=epoch,
                                    group=gi,
                                    groups=len(grouped),
                                    cross_entropy=float(np.mean(recent)),
                                    gradient_norm=gradient,
                                    assistant_targets=supervised_tokens,
                                    input_tokens=input_tokens,
                                    peak_gpu_bytes=torch.cuda.max_memory_allocated()
                                    if device.type == "cuda"
                                    else 0,
                                )
                            if a.max_updates and updates >= a.max_updates:
                                log(
                                    "smoke_complete",
                                    **evaluate(model, valid, a.batch_size, a.context, device),
                                )
                                return
            if a.balanced_dialogues and group_had_loss:
                gradient = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                )
                optimizer.step()
                updates += 1
                if updates == 1 or updates % 100 == 0:
                    log(
                        "train",
                        epoch=epoch,
                        group=gi,
                        groups=len(grouped),
                        cross_entropy=float(np.mean(recent)),
                        gradient_norm=gradient,
                        assistant_targets=supervised_tokens,
                        input_tokens=input_tokens,
                        peak_gpu_bytes=torch.cuda.max_memory_allocated()
                        if device.type == "cuda"
                        else 0,
                        balanced_dialogues=True,
                    )
                if a.max_updates and updates >= a.max_updates:
                    log("smoke_complete", **evaluate(model, valid, a.batch_size, a.context, device))
                    return
            if (gi + 1) % 100 == 0:
                save(epoch, gi + 1)
        score = evaluate(model, valid, a.batch_size, a.context, device)
        log("validation", epoch=epoch, **score)
        improved = score["cross_entropy"] < best
        best = min(best, score["cross_entropy"])
        package = save(epoch + 1, 0, score)
        if improved:
            torch.save(package, a.output / "best.pt")
    atomic_json(
        a.output / "progress.json",
        {
            "status": "complete",
            "epochs": a.epochs,
            "updates": updates,
            "best_cross_entropy": best,
            "initial_cross_entropy": initial["cross_entropy"],
            "elapsed_seconds": time.monotonic() - started,
            "improved_over_initial": best < initial["cross_entropy"],
        },
    )


if __name__ == "__main__":
    main()
