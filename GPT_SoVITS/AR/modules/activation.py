"""Multi-head attention with key/value caching support.

modified from https://github.com/lifeiteng/vall-e/blob/main/valle/modules/activation.py
"""

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Linear, Module
from torch.nn import functional as F
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.parameter import Parameter

from AR.modules.patched_mha_with_cache import multi_head_attention_forward_patched

F.multi_head_attention_forward = multi_head_attention_forward_patched


class MultiheadAttention(Module):
    __constants__ = ["batch_first"]
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        linear1_cls=Linear,
        linear2_cls=Linear,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        if linear1_cls == Linear:
            if not self._qkv_same_embed_dim:
                self.q_proj_weight = Parameter(
                    torch.empty((embed_dim, embed_dim), **factory_kwargs),
                )
                self.k_proj_weight = Parameter(
                    torch.empty((embed_dim, self.kdim), **factory_kwargs),
                )
                self.v_proj_weight = Parameter(
                    torch.empty((embed_dim, self.vdim), **factory_kwargs),
                )
                self.register_parameter("in_proj_weight", None)
            else:
                self.in_proj_weight = Parameter(
                    torch.empty((3 * embed_dim, embed_dim), **factory_kwargs),
                )
                self.register_parameter("q_proj_weight", None)
                self.register_parameter("k_proj_weight", None)
                self.register_parameter("v_proj_weight", None)

            if bias:
                self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
            else:
                self.register_parameter("in_proj_bias", None)
            self.out_proj = NonDynamicallyQuantizableLinear(
                embed_dim,
                embed_dim,
                bias=bias,
                **factory_kwargs,
            )

            self._reset_parameters()
        else:
            if not self._qkv_same_embed_dim:
                raise NotImplementedError
            else:
                self.in_proj_linear = linear1_cls(
                    embed_dim,
                    3 * embed_dim,
                    bias=bias,
                    **factory_kwargs,
                )
                self.in_proj_weight = self.in_proj_linear.weight

                self.register_parameter("q_proj_weight", None)
                self.register_parameter("k_proj_weight", None)
                self.register_parameter("v_proj_weight", None)

                if bias:
                    self.in_proj_bias = self.in_proj_linear.bias
                else:
                    self.register_parameter("in_proj_bias", None)

            self.out_proj = linear2_cls(
                embed_dim,
                embed_dim,
                bias=bias,
                **factory_kwargs,
            )

            if self.bias_k is not None:
                xavier_normal_(self.bias_k)
            if self.bias_v is not None:
                xavier_normal_(self.bias_v)

        self.add_zero_attn = add_zero_attn

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)

        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        cache=None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        is_batched = query.dim() == 3
        if key_padding_mask is not None:
            _kpm_dtype = key_padding_mask.dtype
            if _kpm_dtype != torch.bool and not torch.is_floating_point(
                key_padding_mask,
            ):
                raise AssertionError("only bool and floating types of key_padding_mask are supported")
        why_not_fast_path = ""
        if not is_batched:
            why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        elif query is not key or key is not value:
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = (
                f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
            )
        elif self.in_proj_weight is not None and query.dtype != self.in_proj_weight.dtype:
            why_not_fast_path = (
                f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
            )
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.dropout:
            why_not_fast_path = f"dropout was {self.dropout}, required zero"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif attn_mask is not None:
            why_not_fast_path = "attn_mask was not None"
        elif query.is_nested and key_padding_mask is not None:
            why_not_fast_path = "key_padding_mask is not supported with NestedTensor input"
        elif self.num_heads % 2 == 1:
            why_not_fast_path = "num_heads is odd"
        elif torch.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif not all([(x is None or x.is_cuda or "cpu" in str(x.device)) for x in tensor_args]):
                why_not_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any([x is not None and x.requires_grad for x in tensor_args]):
                why_not_fast_path = "grad is enabled and at least one of query or the input/output projection weights or biases requires_grad"
            if not why_not_fast_path:
                return torch._native_multi_head_attention(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    self.in_proj_weight,
                    self.in_proj_bias,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    key_padding_mask if key_padding_mask is not None else attn_mask,
                    need_weights,
                    average_attn_weights,
                    1 if key_padding_mask is not None else 0 if attn_mask is not None else None,
                )

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, (
            "MultiheadAttention does not support NestedTensor outside of its fast path. "
            + f"The fast path was not hit because {why_not_fast_path}"
        )

        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights,
                cache=cache,
            )
        else:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                cache=cache,
            )
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights
