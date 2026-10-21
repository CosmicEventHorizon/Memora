import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight)


def rope(x, offset=0):
    d = x.shape[-1]
    inv = 1 / (10000 ** (torch.arange(0, d, 2, device=x.device).float() / d))
    pos = torch.arange(offset, offset + x.shape[1], device=x.device).float()
    angle = pos[:, None] * inv[None]
    c, s = angle.cos().to(x.dtype)[None, :, None], angle.sin().to(x.dtype)[None, :, None]
    a, b = x[..., ::2], x[..., 1::2]
    return torch.stack((a * c - b * s, a * s + b * c), -1).flatten(-2)


class SwiGLU(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.up = nn.Linear(width, hidden * 2, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        gate, value = self.up(x).chunk(2, -1)
        return self.down(F.silu(gate) * value)


class CausalSelfAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.dim = width // heads
        if width % heads or self.dim % 2:
            raise ValueError("invalid heads")
        self.qkv = nn.Linear(width, width * 3, bias=False)
        self.output = nn.Linear(width, width, bias=False)

    def forward(self, x):
        b, t, w = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.heads, self.dim).unbind(2)
        q, k = rope(q), rope(k)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        return self.output(y.transpose(1, 2).reshape(b, t, w))


class TransformerBlock(nn.Module):
    def __init__(self, width, heads, hidden):
        super().__init__()
        self.n1 = RMSNorm(width)
        self.attn = CausalSelfAttention(width, heads)
        self.n2 = RMSNorm(width)
        self.ff = SwiGLU(width, hidden)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.ff(self.n2(x))
