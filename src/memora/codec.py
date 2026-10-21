from pathlib import Path

import torch
from moshi.models.loaders import get_mimi
from torch.nn import functional as F

from .paths import MODEL_DIR

DEFAULT_WEIGHTS = MODEL_DIR / "tokenizer-e351c8d8-checkpoint125.safetensors"


class DualTokenizer:
    def __init__(self, device="cuda", weights: Path = DEFAULT_WEIGHTS):
        self.weights = Path(weights)
        self.semantic = get_mimi(self.weights, device=device, num_codebooks=1).eval()
        self.acoustic = get_mimi(self.weights, device=device, num_codebooks=8).eval()
        self.semantic.requires_grad_(False)
        self.acoustic.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, waveform):
        semantic = self.semantic.encode(waveform)[:, 0]
        all_acoustic = self.acoustic.encode(waveform)
        return semantic, all_acoustic[:, 1:]

    @torch.inference_mode()
    def decode(self, semantic, acoustic):
        return self.acoustic.decode(torch.cat((semantic[:, None], acoustic), 1))

    @torch.inference_mode()
    def acoustic_codebook_vectors(self):
        quantizer = self.acoustic.quantizer.rvq_rest
        projection = quantizer.output_proj.weight[:, :, 0]
        tables = []
        for layer in quantizer.vq.layers[: quantizer.n_q]:
            tables.append(F.linear(layer._codebook.embedding, projection))
        return torch.stack(tables)
