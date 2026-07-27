"""Attention module with optional flash attention. einops notation: b batch, h heads, n/i/j sequence length, d feature dimension."""

from packaging import version
import torch
from torch import nn, einsum
import torch.nn.functional as F


def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


class Attend(nn.Module):
    def __init__(self, dropout=0.0, flash=False, scale=None):
        super().__init__()
        self.scale = scale
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse("2.0.0")), (
            "in order to use flash attention, you must be using pytorch 2.0 or above"
        )

    def flash_attn(self, q, k, v):
        if exists(self.scale):
            default_scale = q.shape[-1] ** -0.5
            q = q * (self.scale / default_scale)

        return F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)

    def forward(self, q, k, v):
        scale = default(self.scale, q.shape[-1] ** -0.5)

        if self.flash:
            return self.flash_attn(q, k, v)

        sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        return out
