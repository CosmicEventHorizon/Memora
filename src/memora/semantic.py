from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .common import RMSNorm, SwiGLU, rope


@dataclass(frozen=True)
class MemoraConfig:
    vocabulary: int = 2048
    width: int = 768
    layers: int = 12
    heads: int = 12
    memory_dim: int = 64
    memory_expansion: int = 4
    ff_size: int = 2048
    chunk: int = 16
    context: int = 512


@dataclass
class MemoraState:
    weights: list[tuple[torch.Tensor, torch.Tensor]]
    tokens: int = 0

    def detached(self):
        return MemoraState([(a.detach(), b.detach()) for a, b in self.weights], self.tokens)


def gelu_derivative(x):
    root = 0.7978845608028654
    inner = root * (x + 0.044715 * x.pow(3))
    tanh = inner.tanh()
    return 0.5 * (1 + tanh) + 0.5 * x * (1 - tanh.square()) * root * (1 + 0.134145 * x.square())


def layer_norm_and_backward(x, upstream=None, eps=1e-5):
    mean = x.mean(-1, keepdim=True)
    centered = x - mean
    inv = (centered.square().mean(-1, keepdim=True) + eps).rsqrt()
    normalized = centered * inv
    if upstream is None:
        return normalized
    grad = (
        upstream
        - upstream.mean(-1, keepdim=True)
        - normalized * (upstream * normalized).mean(-1, keepdim=True)
    ) * inv
    return normalized, grad


class MemoraMixer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        h, d, f = c.heads, c.memory_dim, c.memory_dim * c.memory_expansion
        self.norm = RMSNorm(c.width)
        self.qkv = nn.Linear(c.width, h * d * 3, bias=False)
        self.out = nn.Linear(h * d, c.width, bias=False)
        self.out_gate = nn.Linear(c.width, c.width, bias=False)
        self.causal_conv = nn.Conv1d(h * d * 3, h * d * 3, 4, groups=h * d * 3, bias=False)
        self.rate_gate = nn.Sequential(nn.Linear(c.width, 64), nn.SiLU(), nn.Linear(64, h * 2))
        w1 = torch.zeros(h, f, d)
        w2 = torch.zeros(h, d, f)
        for head in range(h):
            w1[head, torch.arange(f), torch.arange(f) % d] = 2
            w2[head, torch.arange(d), torch.arange(d)] = 2
        self.base_w1 = nn.Parameter(w1)
        self.base_w2 = nn.Parameter(w2)

    def initial(self, batch, device, dtype):
        w1 = (
            self.base_w1.softmax(-1)
            .to(device=device, dtype=dtype)
            .expand(batch, -1, -1, -1)
            .clone()
        )
        w2 = (
            self.base_w2.softmax(-1)
            .to(device=device, dtype=dtype)
            .expand(batch, -1, -1, -1)
            .clone()
        )
        return w1, w2

    @staticmethod
    def memory(weight1, weight2, x):
        pre = torch.einsum("bhfd,bchd->bchf", weight1, x)
        hidden = F.gelu(pre)
        raw = torch.einsum("bhdf,bchf->bchd", weight2, hidden)
        return x + layer_norm_and_backward(raw), pre, hidden, raw

    def update(self, w1, w2, key, value, context):
        prediction, pre, hidden, raw = self.memory(w1, w2, key)
        error = (prediction - value).clamp(-5, 5)
        _, delta = layer_norm_and_backward(raw, error)
        grad2 = torch.einsum("bchd,bchf->bhdf", delta, hidden) / key.shape[1]
        delta_hidden = torch.einsum("bhdf,bchd->bchf", w2, delta) * gelu_derivative(pre)
        grad1 = torch.einsum("bchf,bchd->bhfd", delta_hidden, key) / key.shape[1]
        gates = self.rate_gate(context.mean(1)).view(len(context), self.c.heads, 2)
        alpha = 0.95 + 0.049 * torch.sigmoid(gates[..., 0])
        eta = 0.03 * torch.sigmoid(gates[..., 1])
        w1 = torch.softmax(
            alpha[..., None, None] * w1.clamp_min(1e-7).log() - eta[..., None, None] * grad1, -1
        )
        w2 = torch.softmax(
            alpha[..., None, None] * w2.clamp_min(1e-7).log() - eta[..., None, None] * grad2, -1
        )
        return w1, w2, error.square().mean()

    def forward(self, x, state=None):
        b, t, _ = x.shape
        h, d = self.c.heads, self.c.memory_dim
        values = self.qkv(self.norm(x)).transpose(1, 2)
        values = self.causal_conv(F.pad(values, (3, 0))).transpose(1, 2).view(b, t, 3, h, d)
        q, k, v = values.unbind(2)
        q, k = F.normalize(rope(q), dim=-1), F.normalize(rope(k), dim=-1)
        w1, w2 = self.initial(b, x.device, x.dtype) if state is None else state
        outputs = []
        losses = []
        for start in range(0, t, self.c.chunk):
            stop = min(t, start + self.c.chunk)
            qc, kc, vc = q[:, start:stop], k[:, start:stop], v[:, start:stop]
            read, *_ = self.memory(w1, w2, qc)
            outputs.append(read)
            w1, w2, loss = self.update(w1, w2, kc, vc, x[:, start:stop])
            losses.append(loss)
        result = torch.cat(outputs, 1).reshape(b, t, h * d)
        result = self.out(result) * torch.sigmoid(self.out_gate(x))
        return result, (w1, w2), torch.stack(losses).mean()


class MemoraBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.memory = MemoraMixer(c)
        self.ff_norm = RMSNorm(c.width)
        self.ff = SwiGLU(c.width, c.ff_size)

    def forward(self, x, state=None):
        update, new_state, loss = self.memory(x, state)
        x = x + update
        x = x + self.ff(self.ff_norm(x))
        return x, new_state, loss


class MemoraLanguageModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or MemoraConfig()
        c = self.config
        self.embedding = nn.Embedding(c.vocabulary, c.width)
        self.blocks = nn.ModuleList(MemoraBlock(c) for _ in range(c.layers))
        self.norm = RMSNorm(c.width)
        self.bias = nn.Parameter(torch.zeros(c.vocabulary))
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def initial_state(self, batch, device=None, dtype=None):
        p = self.embedding.weight
        device = device or p.device
        dtype = dtype or p.dtype
        return MemoraState([block.memory.initial(batch, device, dtype) for block in self.blocks])

    def forward(self, tokens, state=None, retain=False):
        if tokens.shape[1] > self.config.context:
            tokens = tokens[:, -self.config.context :]
        x = self.embedding(tokens)
        states = []
        losses = []
        existing = state.weights if state is not None else [None] * len(self.blocks)
        for block, old in zip(self.blocks, existing):
            x, new, loss = block(x, old)
            states.append(new)
            losses.append(loss)
        logits = F.linear(self.norm(x), self.embedding.weight, self.bias)
        new_state = MemoraState(states, (state.tokens if state else 0) + tokens.shape[1])
        return {
            "logits": logits,
            "state": new_state if retain else new_state.detached(),
            "memory_loss": torch.stack(losses).mean(),
        }

    def package(self, **meta):
        return {
            "format": 1,
            "architecture": "memora-kl-mlp",
            "config": asdict(self.config),
            "state_dict": self.state_dict(),
            **meta,
        }

    @classmethod
    def from_package(cls, p, device="cpu"):
        m = cls(MemoraConfig(**p["config"])).to(device)
        m.load_state_dict(p["state_dict"])
        return m
