import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

import pyarrow.parquet as pq

from memora.dialogue import atomic_json
from memora.paths import corpus_dir

SPACE = re.compile(r"\s+")
BAD = re.compile(r"```|https?://|www\.|<html|</|\$\$|\\(?:frac|begin)", re.I)
GREETING = re.compile(r"\b(hello|hi|hey|how are you|good morning|good evening)\b", re.I)


def clean(value):
    value = SPACE.sub(" ", str(value).replace("\x00", " ")).strip()
    value = re.sub(r"^(?:Persona [AB]|[A-Z][A-Za-z ]{0,20}):\s*", "", value)
    words = re.findall(r"\b[\w']+\b", value)
    letters = [character for character in value if character.isalpha()]
    if not 2 <= len(words) <= 50 or len(value) > 350 or BAD.search(value):
        return None
    if not letters or sum(character.isascii() for character in letters) / len(letters) < 0.98:
        return None
    return value


def normalize(turns, source):
    turns = list(turns)[:12]
    if len(turns) % 2:
        turns = turns[:-1]
    if len(turns) < 4:
        return None
    cleaned = [clean(turn) for turn in turns]
    if any(turn is None for turn in cleaned):
        return None
    result = [
        {"role": "user" if index % 2 == 0 else "assistant", "text": text}
        for index, text in enumerate(cleaned)
    ]
    assistant_words = [len(turn["text"].split()) for turn in result if turn["role"] == "assistant"]
    if max(assistant_words) > 50 or sum(assistant_words) > 180:
        return None
    digest = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return {
        "id": digest,
        "source": source,
        "turns": result,
        "has_greeting": any(GREETING.search(turn["text"]) for turn in result),
    }


def batches(path, columns=None):
    yield from pq.ParquetFile(path).iter_batches(batch_size=4096, columns=columns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=corpus_dir())
    parser.add_argument("--sources", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    sources = (args.sources or root / "sources").resolve()
    root.mkdir(parents=True, exist_ok=True)
    candidates = {}

    def add(item):
        if item is not None:
            candidates.setdefault(item["id"], item)

    path = sources / "Estwld--empathetic_dialogues_llm/data/train-00000-of-00001.parquet"
    for batch in batches(path, ["conversations"]):
        for row in batch.to_pylist():
            conversation = row["conversations"]
            if isinstance(conversation, str):
                conversation = ast.literal_eval(conversation)
            add(normalize([turn["content"] for turn in conversation], "empathetic_dialogues"))

    path = sources / "Cynaptics--persona-chat/data/train-00000-of-00001.parquet"
    for batch in batches(path, ["dialogue", "reference"]):
        for row in batch.to_pylist():
            turns = list(row["dialogue"])
            if row.get("reference"):
                turns.append(row["reference"])
            add(normalize(turns, "persona_chat"))

    path = sources / "DeepPavlov--daily_dialog/data/train-00000-of-00001.parquet"
    longest = {}
    for batch in batches(path, ["dialog"]):
        for row in batch.to_pylist():
            turns = row["dialog"]
            key = tuple(turns[:2])
            if len(turns) > len(longest.get(key, ())):
                longest[key] = turns
    for turns in longest.values():
        add(normalize(turns, "daily_dialog"))

    soda = []
    path = sources / "allenai--soda/train.parquet"
    for batch in batches(path, ["dialogue"]):
        for row in batch.to_pylist():
            item = normalize(row["dialogue"], "soda")
            if item is not None:
                soda.append(item)
    soda.sort(key=lambda row: row["id"])
    greeting = [row for row in soda if row["has_greeting"]][:35_000]
    selected_ids = {row["id"] for row in greeting}
    other = [row for row in soda if row["id"] not in selected_ids][:115_000]
    for item in greeting + other:
        add(item)

    rows = sorted(candidates.values(), key=lambda row: row["id"])
    output = root / "source-dialogues.jsonl"
    with output.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    words = [len(turn["text"].split()) for row in rows for turn in row["turns"]]
    assistant = [
        len(turn["text"].split())
        for row in rows
        for turn in row["turns"]
        if turn["role"] == "assistant"
    ]
    report = {
        "status": "complete",
        "conversations": len(rows),
        "turns": len(words),
        "estimated_hours_at_150_wpm": sum(words) / 150 / 60,
        "greeting_conversation_fraction": sum(row["has_greeting"] for row in rows) / len(rows),
        "mean_assistant_words": sum(assistant) / len(assistant),
        "maximum_assistant_words": max(assistant),
        "sources": {
            source: sum(row["source"] == source for row in rows)
            for source in sorted({row["source"] for row in rows})
        },
    }
    atomic_json(root / "SOURCES_COMPLETE.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
