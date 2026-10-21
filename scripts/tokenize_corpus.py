import argparse
import fcntl
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from memora.codec import DualTokenizer
from memora.dialogue import atomic_json, read_rows, repair_jsonl_tail
from memora.paths import corpus_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=corpus_dir())
    p.add_argument("--follow", action="store_true")
    a = p.parse_args()
    root = a.root.resolve()
    lock = (root / "tokenization.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    repair_jsonl_tail(root / "tokens.jsonl")
    existing = read_rows(root / "tokens.jsonl")
    done = {r["id"] for r in existing}
    codec = DualTokenizer()
    started = time.monotonic()
    new_seconds = 0
    while True:
        pending = [r for r in read_rows(root / "audio.jsonl") if r["id"] not in done]
        for row in pending:
            output = []
            for turn in row["turns"]:
                path = (
                    root
                    / "tokens"
                    / row["split"]
                    / row["id"][:2]
                    / row["id"]
                    / f"{turn['index']:02d}.npz"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    audio, rate = sf.read(turn["audio"], dtype="float32")
                    assert rate == 24000 and audio.ndim == 1
                    count = (len(audio) + 1919) // 1920
                    padded = np.pad(audio, (0, count * 1920 - len(audio)))
                    semantic, acoustic = codec.encode(torch.from_numpy(padded)[None, None].cuda())
                    s = semantic[0, :count].cpu().numpy().astype("<u2")
                    ac = acoustic[0, :, :count].cpu().numpy().astype("<u2")
                    assert (
                        s.shape == (count,)
                        and ac.shape == (7, count)
                        and s.max() < 2048
                        and ac.max() < 2048
                    )
                    tmp = path.with_suffix(".partial.npz")
                    np.savez_compressed(tmp, semantic=s, acoustic=ac)
                    tmp.replace(path)
                with np.load(path) as data:
                    count = len(data["semantic"])
                    if data["acoustic"].shape != (7, count):
                        raise ValueError("Token shape mismatch")
                output.append(dict(turn, tokens=str(path), frames=count))
            encoded = dict(row, turns=output)
            with (root / "tokens.jsonl").open("a") as out:
                out.write(json.dumps(encoded) + "\n")
            done.add(row["id"])
            new_seconds += row["seconds"]
            status = {
                "status": "running",
                "conversations": len(done),
                "new_audio_hours": new_seconds / 3600,
                "x_realtime": new_seconds / max(0.001, time.monotonic() - started),
            }
            atomic_json(root / "tokenization-progress.json", status)
            if len(done) % 10 == 0:
                print(json.dumps(status), flush=True)
        if (root / "AUDIO_COMPLETE.json").exists() and len(done) == len(
            read_rows(root / "audio.jsonl")
        ):
            break
        if not a.follow:
            return
        time.sleep(30)
    rows = read_rows(root / "tokens.jsonl")
    acoustic_root = root / "acoustic"
    acoustic_root.mkdir(exist_ok=True)
    report = {"conversations": len(rows), "splits": {}, "assistant_only_acoustic": True}
    for split in ("train", "valid"):
        selected = [r for r in rows if r["split"] == split]
        atomic_json(root / f"{split}.json", selected)
        turns = [
            dict(t, id=f"{r['id']}-{t['index']}", source="tts", conversation_id=r["id"])
            for r in selected
            for t in r["turns"]
            if t["role"] == "assistant"
        ]
        atomic_json(acoustic_root / f"{split}.json", turns)
        report["splits"][split] = {
            "conversations": len(selected),
            "hours": sum(r["seconds"] for r in selected) / 3600,
            "assistant_hours": sum(t["seconds"] for t in turns) / 3600,
        }
    if not rows or not report["splits"]["valid"]["conversations"]:
        raise ValueError("Missing held-out conversations")
    atomic_json(root / "TOKENS_COMPLETE.json", report)
    atomic_json(
        root / "tokenization-progress.json",
        {
            "status": "complete",
            "conversations": len(rows),
            "hours": sum(r["seconds"] for r in rows) / 3600,
        },
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
