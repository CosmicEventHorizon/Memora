import json
import random
from pathlib import Path

import numpy as np
import torch


class TokenWindows:
    def __init__(self, root, split, context, source=None, limit=None):
        self.rows = json.loads((Path(root) / f"{split}.json").read_text())
        self.context = context
        if source:
            self.rows = [r for r in self.rows if r["source"] == source]
        if limit:
            self.rows = self.rows[:limit]
        self.windows = []
        for i, row in enumerate(self.rows):
            for start in range(0, max(1, row["frames"] - 1), context):
                length = min(context, row["frames"] - start - 1)
                if length > 0:
                    self.windows.append((i, start, length))

    def batch(self, indices, device="cuda"):
        selected = [self.windows[int(i)] for i in indices]
        length = max(x[2] for x in selected)
        semantic = torch.zeros(len(selected), length + 1, dtype=torch.long, device=device)
        acoustic = torch.zeros(len(selected), 7, length + 1, dtype=torch.long, device=device)
        valid = torch.zeros(len(selected), length, dtype=torch.bool, device=device)
        for b, (row_i, start, count) in enumerate(selected):
            with np.load(self.rows[row_i]["tokens"]) as item:
                semantic[b, : count + 1] = torch.from_numpy(
                    item["semantic"][start : start + count + 1].astype("i8")
                ).to(device)
                acoustic[b, :, : count + 1] = torch.from_numpy(
                    item["acoustic"][:, start : start + count + 1].astype("i8")
                ).to(device)
            semantic[b, count + 1 :] = semantic[b, count]
            acoustic[b, :, count + 1 :] = acoustic[b, :, count, None]
            valid[b, :count] = True
        return semantic, acoustic, valid

    def order(self, seed):
        order = list(range(len(self.windows)))
        random.Random(seed).shuffle(order)
        return order
