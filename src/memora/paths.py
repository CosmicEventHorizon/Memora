from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RUN_DIR = PROJECT_ROOT / "runs"
LOG_DIR = PROJECT_ROOT / "logs"


def corpus_dir() -> Path:
    return DATA_DIR / "conversations"
