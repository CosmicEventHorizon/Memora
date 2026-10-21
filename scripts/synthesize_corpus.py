import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

from memora.dialogue import atomic_json, read_rows, repair_jsonl_tail, split_for
from memora.paths import MODEL_DIR, corpus_dir
from memora.tts import load_voice, synthesize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=corpus_dir())
    parser.add_argument("--hours", type=float, default=1_300)
    parser.add_argument(
        "--assistant-voice",
        type=Path,
        default=MODEL_DIR / "assistant.onnx",
    )
    parser.add_argument("--user-voice", type=Path, action="append", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    root = args.root.resolve()
    rows = read_rows(root / "source-dialogues.jsonl")
    if not rows:
        raise ValueError("Run prepare_corpus.py first")
    repair_jsonl_tail(root / "audio.jsonl")
    existing = read_rows(root / "audio.jsonl")
    done = {row["id"] for row in existing}
    seconds = sum(row["seconds"] for row in existing)
    paths = {"assistant": args.assistant_voice.resolve()}
    paths.update({f"user-{index}": path.resolve() for index, path in enumerate(args.user_voice)})
    voices = {name: load_voice(path, device=args.device) for name, path in paths.items()}
    started = time.monotonic()
    initial = seconds

    def status(value):
        elapsed = time.monotonic() - started
        report = {
            "status": value,
            "conversations": len(done),
            "hours": seconds / 3600,
            "target_hours": args.hours,
            "x_realtime": (seconds - initial) / max(elapsed, 0.001),
            "tts_device": args.device,
        }
        atomic_json(root / "generation-progress.json", report)
        print(json.dumps(report), flush=True)

    if seconds >= args.hours * 3600:
        status("complete")
        atomic_json(root / "AUDIO_COMPLETE.json", {"status": "complete", "hours": seconds / 3600})
        return
    status("running")
    for row in rows:
        if row["id"] in done:
            continue
        split = split_for(row["id"])
        user_voice = f"user-{int(row['id'][:8], 16) % len(args.user_voice)}"
        folder = root / "audio" / split / row["id"][:2] / row["id"]
        folder.mkdir(parents=True, exist_ok=True)
        turns = []
        for index, turn in enumerate(row["turns"]):
            while shutil.disk_usage(root).free < 15 * 1024**3:
                status("waiting_for_disk_space")
                time.sleep(60)
            voice_name = "assistant" if turn["role"] == "assistant" else user_voice
            path = folder / f"{index:02d}-{turn['role']}.flac"
            if not path.exists():
                ort.set_seed((42 + int(row["id"][:8], 16) + index) % (2**31 - 1))
                waveform = synthesize(voices[voice_name], turn["text"].replace("\x00", " "))
                waveform = np.pad(waveform, (0, int(0.2 * 24000)))
                temporary = path.with_suffix(".partial.flac")
                sf.write(temporary, waveform, 24000, subtype="PCM_16", format="FLAC")
                temporary.replace(path)
            info = sf.info(path)
            turns.append(
                {
                    **turn,
                    "voice": voice_name,
                    "audio": str(path),
                    "seconds": info.frames / 24000,
                    "samples": info.frames,
                    "index": index,
                }
            )
        encoded = {
            "id": row["id"],
            "source": row["source"],
            "split": split,
            "turns": turns,
            "seconds": sum(turn["seconds"] for turn in turns),
        }
        with (root / "audio.jsonl").open("a") as stream:
            stream.write(json.dumps(encoded) + "\n")
            stream.flush()
        done.add(row["id"])
        seconds += encoded["seconds"]
        if len(done) % 10 == 0:
            status("running")
        if seconds >= args.hours * 3600:
            status("complete")
            atomic_json(
                root / "AUDIO_COMPLETE.json",
                {
                    "status": "complete",
                    "hours": seconds / 3600,
                    "conversations": len(done),
                    "target_hours": args.hours,
                    "assistant_voice": str(args.assistant_voice),
                    "sources": sorted({item["source"] for item in rows}),
                },
            )
            return
    raise RuntimeError(f"Casual source corpus exhausted at {seconds / 3600:.1f} hours")


if __name__ == "__main__":
    main()
