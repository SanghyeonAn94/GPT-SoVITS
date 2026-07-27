"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from x_transformers.x_transformers import RotaryEmbedding

from GPT_SoVITS.f5_tts.model.modules import (
    TimestepEmbedding,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    AdaLayerNormZero_Final,
    precompute_freqs_cis,
    get_pos_embed_indices,
)

from module.commons import sequence_mask


class TextEmbedding(nn.Module):
    def __init__(self, text_dim, conv_layers=0, conv_mult=2):
        super().__init__()
        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(self, text: int["b nt"], seq_len, drop_text=False):  # noqa: F722
        batch, text_len = text.shape[0], text.shape[1]

        if drop_text:
            text = torch.zeros_like(text)

        if self.extra_modeling:
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]

            text = text + text_pos_embed

            text = self.text_blocks(text)

        return text


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: float["b n d"], cond: float["b n d"], text_embed: float["b n d"], drop_audio_cond=False):  # noqa: F722
        if drop_audio_cond:
            cond = torch.zeros_like(cond)

        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_dim=None,
        conv_layers=0,
        long_skip_connection=False,
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        self.d_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(text_dim, conv_layers=conv_layers)
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout) for _ in range(depth)]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNormZero_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def forward(
        self,
        x0: float["b n d"],  # noqa: F722
        cond0: float["b n d"],  # noqa: F722
        x_lens,
        time: float["b"] | float[""],  # noqa: F821 F722
        dt_base_bootstrap,
        text0,  # noqa: F722
        use_grad_ckpt=False,
        drop_audio_cond=False,
        drop_text=False,
        infer=False,
        text_cache=None,
        dt_cache=None,
    ):
        x = x0.transpose(2, 1)
        cond = cond0.transpose(2, 1)
        text = text0.transpose(2, 1)
        mask = sequence_mask(x_lens, max_length=x.size(1)).to(x.device)

        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        t = self.time_embed(time)
        if infer and dt_cache is not None:
            dt = dt_cache
        else:
            dt = self.d_embed(dt_base_bootstrap)
        t += dt

        if infer and text_cache is not None:
            text_embed = text_cache
        else:
            text_embed = self.text_embed(text, seq_len, drop_text=drop_text)

        x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            if use_grad_ckpt:
                x = checkpoint(self.ckpt_wrapper(block), x, t, mask, rope, use_reentrant=False)
            else:
                x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        if infer:
            return output, text_embed, dt
        else:
            return output
