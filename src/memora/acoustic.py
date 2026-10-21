from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .common import RMSNorm, TransformerBlock


@dataclass(frozen=True)
class AcousticConfig:
    semantic_vocabulary: int = 2048
    acoustic_vocabulary: int = 2048
    codebooks: int = 7
    acoustic_dimension: int = 512
    width: int = 768
    layers: int = 12
    heads: int = 12
    ff_size: int = 2048
    depth_width: int = 512
    depth_layers: int = 4
    depth_heads: int = 8
    context: int = 512


class AcousticPredictor(nn.Module):
    def __init__(self, config=None, codebook_vectors=None):
        super().__init__()
        self.config = config or AcousticConfig()
        c = self.config
        if codebook_vectors is None:
            codebook_vectors = torch.zeros(c.codebooks, c.acoustic_vocabulary, c.acoustic_dimension)
        expected = (c.codebooks, c.acoustic_vocabulary, c.acoustic_dimension)
        if tuple(codebook_vectors.shape) != expected:
            raise ValueError(
                f"expected codebook vectors {expected}, got {tuple(codebook_vectors.shape)}"
            )
        self.register_buffer("codebook_vectors", codebook_vectors.float().clone())
        self.semantic = nn.Embedding(c.semantic_vocabulary, c.width)
        self.acoustic_input = nn.Linear(c.acoustic_dimension, c.width, bias=False)
        self.blocks = nn.ModuleList(
            TransformerBlock(c.width, c.heads, c.ff_size) for _ in range(c.layers)
        )
        self.norm = RMSNorm(c.width)
        self.to_depth = nn.Linear(c.width, c.depth_width)
        self.codebook_to_depth = nn.Linear(c.acoustic_dimension, c.depth_width, bias=False)
        self.depth_position = nn.Parameter(torch.randn(c.codebooks, c.depth_width) * 0.02)
        self.depth = nn.ModuleList(
            TransformerBlock(c.depth_width, c.depth_heads, c.depth_width * 4)
            for _ in range(c.depth_layers)
        )
        self.depth_norm = RMSNorm(c.depth_width)
        self.heads = nn.ModuleList(
            nn.Linear(c.depth_width, c.acoustic_vocabulary) for _ in range(c.codebooks)
        )
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def acoustic_state(self, tokens):
        return sum(
            F.embedding(tokens[:, q], self.codebook_vectors[q])
            for q in range(self.config.codebooks)
        )

    def time_state(self, semantic, previous_acoustic):
        if semantic.shape[-1] > self.config.context:
            semantic = semantic[:, -self.config.context :]
            previous_acoustic = previous_acoustic[:, :, -self.config.context :]
        previous_state = self.acoustic_state(previous_acoustic)
        x = self.semantic(semantic) + self.acoustic_input(previous_state)
        for block in self.blocks:
            x = block(x)
        return self.to_depth(self.norm(x))

    def depth_logits(self, time_state, current_acoustic):
        b, t, _ = time_state.shape
        teacher = current_acoustic.permute(0, 2, 1).reshape(b * t, self.config.codebooks)
        sequence = [time_state.reshape(b * t, 1, -1)]
        for q in range(self.config.codebooks - 1):
            vector = F.embedding(teacher[:, q], self.codebook_vectors[q])
            sequence.append(self.codebook_to_depth(vector)[:, None])
        y = torch.cat(sequence, 1) + self.depth_position[None]
        for block in self.depth:
            y = block(y)
        y = self.depth_norm(y)
        logits = torch.stack([head(y[:, q]) for q, head in enumerate(self.heads)], 1)
        return logits.view(b, t, self.config.codebooks, self.config.acoustic_vocabulary).permute(
            0, 2, 1, 3
        )

    def forward(self, semantic, previous_acoustic, current_acoustic):
        return self.depth_logits(self.time_state(semantic, previous_acoustic), current_acoustic)

    @torch.inference_mode()
    def generate_frame(self, semantic, acoustic_history, temperature=0.8, top_k=50, generator=None):
        b, t = semantic.shape
        previous = torch.zeros(
            b, self.config.codebooks, t, dtype=torch.long, device=semantic.device
        )
        if t > 1:
            previous[:, :, 1:] = acoustic_history[:, :, -(t - 1) :]
        time_state = self.time_state(semantic, previous)[:, -1]
        chosen = []
        sequence = time_state[:, None]
        for q in range(self.config.codebooks):
            y = sequence + self.depth_position[: sequence.shape[1]][None]
            for block in self.depth:
                y = block(y)
            logits = self.heads[q](self.depth_norm(y)[:, -1]).float()
            if temperature <= 0:
                token = logits.argmax(-1)
            else:
                values, indices = (logits / temperature).topk(min(top_k, logits.shape[-1]), -1)
                pick = torch.multinomial(values.softmax(-1), 1, generator=generator)
                token = indices.gather(-1, pick).squeeze(-1)
            chosen.append(token)
            if q < self.config.codebooks - 1:
                vector = F.embedding(token, self.codebook_vectors[q])
                sequence = torch.cat((sequence, self.codebook_to_depth(vector)[:, None]), 1)
        return torch.stack(chosen, 1)

    def package(self, **meta):
        return {
            "format": 3,
            "architecture": "summed-state-temporal-depth-transformer",
            "config": asdict(self.config),
            "state_dict": self.state_dict(),
            **meta,
        }

    @classmethod
    def from_package(cls, p, device="cpu"):
        if p.get("architecture") != "summed-state-temporal-depth-transformer":
            raise ValueError(
                "Obsolete acoustic checkpoint; retrain the temporal-depth acoustic Transformer"
            )
        model = cls(AcousticConfig(**p["config"])).to(device)
        model.load_state_dict(p["state_dict"])
        return model
