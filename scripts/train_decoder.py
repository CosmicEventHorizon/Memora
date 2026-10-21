import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

from memora.decoder import AssistantDecoder
from memora.paths import MODEL_DIR, corpus_dir


def spectral(pred, target):
    total = pred.new_zeros(())
    for n in (256, 512, 1024):
        window = torch.hann_window(n, device=pred.device)
        a = torch.stft(pred, n, n // 4, window=window, return_complex=True).abs()
        b = torch.stft(target, n, n // 4, window=window, return_complex=True).abs()
        total += F.l1_loss(torch.log1p(a), torch.log1p(b))
    return total / 3


class AssistantData:
    def __init__(self, root, split):
        self.rows = [
            r
            for r in json.loads((Path(root) / f"{split}.json").read_text())
            if r["source"] == "tts"
        ]
        if not self.rows:
            raise ValueError("decoder requires TTS-only rows")

    def sample(self, rng, frames=50):
        row = self.rows[rng.randrange(len(self.rows))]
        with np.load(row["tokens"]) as x:
            semantic = x["semantic"]
            acoustic = x["acoustic"]
        count = min(frames, len(semantic))
        start = rng.randrange(max(1, len(semantic) - count + 1))
        wave, rate = sf.read(row["audio"], dtype="float32")
        assert rate == 24000
        target = wave[start * 1920 : (start + count) * 1920]
        target = np.pad(target, (0, count * 1920 - len(target)))
        return (
            torch.from_numpy(semantic[start : start + count].astype("i8"))[None].cuda(),
            torch.from_numpy(acoustic[:, start : start + count].astype("i8"))[None].cuda(),
            torch.from_numpy(target)[None].cuda(),
        )


@torch.inference_mode()
def evaluate(model, data, steps=20):
    model.eval()
    rng = random.Random(99)
    scores = []
    for _ in range(steps):
        s, a, w = data.sample(rng)
        y = model(s, a)[0, 0, : w.shape[-1]]
        scores.append((float(F.l1_loss(y, w[0])), float(spectral(y, w[0]))))
    model.train()
    return {"wave_l1": float(np.mean(scores, 0)[0]), "spectral": float(np.mean(scores, 0)[1])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=corpus_dir() / "acoustic")
    p.add_argument("--output", type=Path, default=MODEL_DIR / "decoder")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--fresh", action="store_true")
    a = p.parse_args()
    if a.fresh and a.output.exists():
        shutil.rmtree(a.output)
    a.output.mkdir(parents=True, exist_ok=not a.fresh)
    train = AssistantData(a.data, "train")
    valid = AssistantData(a.data, "valid")
    model = AssistantDecoder().cuda()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": [model.semantic.weight, model.semantic_scale], "lr": 2e-4},
            {
                "params": [
                    p
                    for p in params
                    if p is not model.semantic.weight and p is not model.semantic_scale
                ],
                "lr": 1e-5,
            },
        ],
        betas=(0.9, 0.95),
    )
    rng = random.Random(42)
    started = time.monotonic()
    best = math.inf

    def log(kind, **v):
        row = {"kind": kind, "step": step, "seconds": time.monotonic() - started, **v}
        open(a.output / "metrics.jsonl", "a").write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        return row

    step = 0
    log("initial", trainable_parameters=sum(p.numel() for p in params), **evaluate(model, valid))
    for step in range(1, a.steps + 1):
        semantic, acoustic, target = train.sample(rng)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(semantic, acoustic)[0, 0, : target.shape[-1]]
            loss = spectral(prediction.float(), target[0]) + 0.2 * F.l1_loss(
                prediction.float(), target[0]
            )
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(params, 1))
        optimizer.step()
        if step % 100 == 0:
            log(
                "train",
                loss=float(loss.detach()),
                gradient_norm=gradient,
                semantic_scale=float(model.semantic_scale.detach()),
                peak_gpu_bytes=torch.cuda.max_memory_allocated(),
            )
        if step % 1000 == 0:
            score = log("validation", **evaluate(model, valid))
            package = {
                "format": 1,
                "architecture": "assistant-mimi-decoder-adapter",
                "step": step,
                "state": model.trainable_state(),
                "validation": score,
            }
            torch.save(package, a.output / "checkpoint.pt")
            if score["spectral"] < best:
                best = score["spectral"]
                torch.save(package, a.output / "best.pt")
    (a.output / "progress.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "steps": a.steps,
                "best_spectral": best,
                "elapsed_seconds": time.monotonic() - started,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
