import argparse
import json
import subprocess
import sys
from pathlib import Path

from memora.dialogue import atomic_json
from memora.paths import LOG_DIR, MODEL_DIR, PROJECT_ROOT, RUN_DIR, corpus_dir


def run(stage: str, command: list[str], status_path: Path) -> None:
    atomic_json(status_path, {"status": "running", "stage": stage})
    try:
        subprocess.run([sys.executable, *command], cwd=PROJECT_ROOT, check=True)
    except BaseException as error:
        atomic_json(status_path, {"status": "failed", "stage": stage, "error": str(error)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=corpus_dir())
    parser.add_argument("--hours", type=float, default=1_300)
    parser.add_argument("--semantic-epochs", type=int, default=5)
    parser.add_argument("--acoustic-epochs", type=int, default=5)
    parser.add_argument("--decoder-steps", type=int, default=10_000)
    parser.add_argument("--assistant-voice", type=Path, required=True)
    parser.add_argument("--user-voice", type=Path, action="append", required=True)
    args = parser.parse_args()

    corpus = args.corpus.resolve()
    status = LOG_DIR / "pipeline.json"
    scripts = PROJECT_ROOT / "scripts"
    semantic = MODEL_DIR / "semantic"
    acoustic = MODEL_DIR / "acoustic"
    decoder = MODEL_DIR / "decoder"

    run("prepare", [str(scripts / "prepare_corpus.py"), "--root", str(corpus)], status)
    synthesis = [
        str(scripts / "synthesize_corpus.py"),
        "--root",
        str(corpus),
        "--hours",
        str(args.hours),
        "--assistant-voice",
        str(args.assistant_voice),
    ]
    for voice in args.user_voice:
        synthesis.extend(("--user-voice", str(voice)))
    run("synthesize", synthesis, status)
    run("tokenize", [str(scripts / "tokenize_corpus.py"), "--root", str(corpus)], status)
    run(
        "semantic",
        [
            str(scripts / "train_semantic.py"),
            "--data",
            str(corpus),
            "--output",
            str(semantic),
            "--epochs",
            str(args.semantic_epochs),
            "--resume",
        ],
        status,
    )
    run(
        "acoustic",
        [
            str(scripts / "train_acoustic.py"),
            "--data",
            str(corpus / "acoustic"),
            "--output",
            str(acoustic),
            "--epochs",
            str(args.acoustic_epochs),
            "--resume",
        ],
        status,
    )
    run(
        "decoder",
        [
            str(scripts / "train_decoder.py"),
            "--data",
            str(corpus / "acoustic"),
            "--output",
            str(decoder),
            "--steps",
            str(args.decoder_steps),
        ],
        status,
    )
    run(
        "generate",
        [
            str(scripts / "generate_dialogue.py"),
            "--semantic-checkpoint",
            str(semantic / "best.pt"),
            "--acoustic-checkpoint",
            str(acoustic / "best.pt"),
            "--decoder-checkpoint",
            str(decoder / "best.pt"),
            "--output",
            str(RUN_DIR / "conversation"),
        ],
        status,
    )
    report = {"status": "complete", "stage": "complete"}
    atomic_json(status, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
