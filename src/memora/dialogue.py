import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .semantic import MemoraLanguageModel

AUDIO_VOCABULARY = 2048
CONTROL_TOKENS = {"<user>": 2048, "<assistant>": 2049, "<end_turn>": 2050}
VOCABULARY = 2051


def split_for(conversation_id):
    return (
        "valid"
        if int(hashlib.sha256(conversation_id.encode()).hexdigest()[:8], 16) % 20 == 0
        else "train"
    )


def expand_vocabulary(model):
    if model.config.vocabulary == VOCABULARY:
        return model
    if model.config.vocabulary != AUDIO_VOCABULARY:
        raise ValueError("Unexpected base vocabulary")
    parameter = model.embedding.weight
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        expanded = MemoraLanguageModel(replace(model.config, vocabulary=VOCABULARY))
    expanded = expanded.to(device=parameter.device, dtype=parameter.dtype)
    old = model.state_dict()
    with torch.no_grad():
        for name, target in expanded.state_dict().items():
            if name in ("embedding.weight", "bias"):
                target[:AUDIO_VOCABULARY].copy_(old[name])
            else:
                target.copy_(old[name])
        expanded.bias[AUDIO_VOCABULARY:].zero_()
    return expanded


def semantic_sequence(turns):
    tokens, supervised = [], []
    for turn in turns:
        role = turn["role"]
        if role not in ("user", "assistant"):
            raise ValueError(role)
        audio = np.asarray(turn["semantic"], dtype=np.int64).reshape(-1)
        if not len(audio) or audio.min() < 0 or audio.max() >= AUDIO_VOCABULARY:
            raise ValueError("Invalid speech token IDs")
        tokens.extend(
            [CONTROL_TOKENS[f"<{role}>"]] + audio.tolist() + [CONTROL_TOKENS["<end_turn>"]]
        )
        supervised.extend([False] + [role == "assistant"] * (len(audio) + 1))
    return np.asarray(tokens, dtype=np.int64), np.asarray(supervised, dtype=bool)


def load_semantic_sequence(row):
    turns = []
    for turn in row["turns"]:
        with np.load(turn["tokens"]) as data:
            turns.append({"role": turn["role"], "semantic": data["semantic"]})
    return semantic_sequence(turns)


def windows(tokens, mask, context):
    for start in range(0, len(tokens) - 1, context):
        stop = min(start + context, len(tokens) - 1)
        yield tokens[start:stop], tokens[start + 1 : stop + 1], mask[start + 1 : stop + 1]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_rows(path):
    if not Path(path).exists():
        return []
    lines = Path(path).read_text().splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.endswith("\n") and line.strip()]


def repair_jsonl_tail(path):
    path = Path(path)
    if path.exists():
        data = path.read_bytes()
        if data and not data.endswith(b"\n"):
            with path.open("r+b") as stream:
                stream.truncate(data.rfind(b"\n") + 1)


def reply_logits(logits):
    if logits.shape[-1] != VOCABULARY:
        raise ValueError("Expected extended semantic vocabulary")
    result = logits.clone()
    result[..., CONTROL_TOKENS["<user>"]] = float("-inf")
    result[..., CONTROL_TOKENS["<assistant>"]] = float("-inf")
    return result
