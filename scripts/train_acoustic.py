import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from memora.acoustic import AcousticConfig, AcousticPredictor
from memora.codec import DualTokenizer
from memora.data import TokenWindows
from memora.paths import MODEL_DIR, corpus_dir


def loss(model, semantic, acoustic, valid):
    target = acoustic[:, :, 1:]
    logits = model(semantic[:, 1:], acoustic[:, :, :-1], target)
    mask = valid[:, None].expand_as(target)
    ce = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none"
    ).view_as(target)
    classification = (ce * mask).sum() / mask.sum()
    return classification, logits, target, mask


@torch.inference_mode()
def evaluate(model, data, batch, maximum=100_000):
    model.eval()
    cross_entropy = correct = count = 0.0
    code_loss = torch.zeros(model.config.codebooks, device="cuda")
    code_count = torch.zeros_like(code_loss)
    code_correct = torch.zeros_like(code_loss)
    batches = 0
    for start in range(0, len(data.windows), batch):
        semantic, acoustic, valid = data.batch(range(start, min(start + batch, len(data.windows))))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            classification, logits, target, mask = loss(model, semantic, acoustic, valid)
        ce = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none"
        ).view_as(target)
        predicted = logits.argmax(-1)
        targets = float(mask.sum())
        cross_entropy += float(classification) * targets
        correct += float(((predicted == target) & mask).sum())
        count += targets
        code_loss += (ce * mask).sum((0, 2))
        code_count += mask.sum((0, 2))
        code_correct += ((predicted == target) & mask).sum((0, 2))
        batches += 1
        if count >= maximum:
            break
    model.train()
    ce = cross_entropy / count
    return {
        "cross_entropy": ce,
        "perplexity": math.exp(ce),
        "top1": correct / count,
        "targets": int(count),
        "batches": batches,
        "codebook_cross_entropy": (code_loss / code_count).tolist(),
        "codebook_top1": (code_correct / code_count).tolist(),
    }


def module_norms(model):
    groups = {"backbone": [], "heads": []}
    for name, parameter in model.named_parameters():
        group = "heads" if name.startswith("heads.") else "backbone"
        groups[group].append(parameter)

    def norm(items, gradient=False):
        values = [
            (p.grad if gradient else p).detach().float().square().sum()
            for p in items
            if not gradient or p.grad is not None
        ]
        return math.sqrt(float(torch.stack(values).sum())) if values else 0.0

    weights = {key: norm(value) for key, value in groups.items()}
    gradients = {key: norm(value, True) for key, value in groups.items()}
    return gradients, {key: gradients[key] / max(weights[key], 1e-12) for key in groups}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=corpus_dir() / "acoustic")
    parser.add_argument("--output", type=Path, default=MODEL_DIR / "acoustic")
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    args = parser.parse_args()
    if args.fresh:
        if args.output.exists():
            shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=not args.fresh)
    train = TokenWindows(args.data, "train", args.context, limit=args.limit)
    validation = TokenWindows(
        args.data, "valid", args.context, limit=max(8, args.limit // 10) if args.limit else None
    )
    torch.manual_seed(42)
    checkpoint = args.output / "checkpoint.pt"
    resumed = args.resume and checkpoint.exists()
    if checkpoint.exists() and not args.resume:
        raise ValueError("Use --resume or --fresh for an existing checkpoint")
    if resumed:
        package = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = AcousticPredictor.from_package(package)
    elif args.initial:
        package = torch.load(args.initial, map_location="cpu", weights_only=True)
        model = AcousticPredictor.from_package(package)
    else:
        tokenizer = DualTokenizer("cpu")
        vectors = tokenizer.acoustic_codebook_vectors()
        del tokenizer
        config = AcousticConfig(
            width=args.width,
            layers=args.layers,
            heads=max(1, args.width // 64),
            ff_size=int(args.width * 8 / 3),
            context=args.context,
        )
        model = AcousticPredictor(config, vectors)
    model = model.cuda().to(torch.bfloat16)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1
    )
    if resumed:
        optimizer.load_state_dict(package["optimizer"])
    updates = package.get("updates", 0) if resumed else 0
    best = package.get("best_cross_entropy", math.inf) if resumed else math.inf
    started = time.monotonic()
    steps_per_epoch = math.ceil(len(train.windows) / args.batch_size)
    resume_updates = updates

    def log(kind, **values):
        row = {"kind": kind, "updates": updates, "seconds": time.monotonic() - started, **values}
        open(args.output / "metrics.jsonl", "a").write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        return row

    log(
        "initial",
        architecture="summed-state-temporal-depth-transformer",
        parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        windows=len(train.windows),
        **evaluate(model, validation, args.batch_size),
    )
    for epoch in range(resume_updates // steps_per_epoch + 1, args.epochs + 1):
        order = train.order(142 + epoch)
        recent = []
        for start in range(0, len(order), args.batch_size):
            if (epoch - 1) * steps_per_epoch + start // args.batch_size < resume_updates:
                continue
            semantic, acoustic, valid = train.batch(order[start : start + args.batch_size])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                classification, *_ = loss(model, semantic, acoustic, valid)
            if not torch.isfinite(classification):
                raise FloatingPointError(f"nonfinite loss at update {updates}")
            classification.backward()
            gradient = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            )
            diagnostics = module_norms(model) if (updates + 1) % 50 == 0 or updates == 0 else None
            optimizer.step()
            updates += 1
            recent.append(float(classification.detach()))
            if updates % 50 == 0 or updates == 1:
                mean = np.mean(recent[-50:])
                by_module, relative = diagnostics
                log(
                    "train",
                    epoch=epoch,
                    cross_entropy=float(mean),
                    gradient_norm=gradient,
                    gradient_by_module=by_module,
                    gradient_to_weight=relative,
                    peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                    updates_per_second=(updates - resume_updates) / (time.monotonic() - started),
                )
            if updates % 500 == 0:
                torch.save(
                    model.package(
                        epoch=epoch,
                        updates=updates,
                        optimizer=optimizer.state_dict(),
                        best_cross_entropy=best,
                    ),
                    checkpoint,
                )
            if args.max_updates and updates >= args.max_updates:
                score = log("short_run_complete", **evaluate(model, validation, args.batch_size))
                package = model.package(
                    epoch=epoch,
                    updates=updates,
                    validation=score,
                    optimizer=optimizer.state_dict(),
                    best_cross_entropy=min(best, score["cross_entropy"]),
                )
                torch.save(package, checkpoint)
                if score["cross_entropy"] < best or not (args.output / "best.pt").exists():
                    torch.save(package, args.output / "best.pt")
                return
        score = log("validation", epoch=epoch, **evaluate(model, validation, args.batch_size))
        package = model.package(
            epoch=epoch,
            updates=updates,
            validation=score,
            optimizer=optimizer.state_dict(),
            best_cross_entropy=min(best, score["cross_entropy"]),
        )
        torch.save(package, checkpoint)
        if score["cross_entropy"] < best:
            best = score["cross_entropy"]
            torch.save(package, args.output / "best.pt")
    (args.output / "progress.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "epochs": args.epochs,
                "best_cross_entropy": best,
                "elapsed_seconds": time.monotonic() - started,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
