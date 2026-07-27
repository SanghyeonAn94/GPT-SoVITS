# Copyright    2022  Xiaomi Corp.        (authors: Daniel Povey)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Scaling and activation utilities.

DoubleSwish: double_swish(x) = x * sigmoid(x - 1), a close numerical
approximation to swish(swish(x)) where swish(x) = x * sigmoid(x). Its derivative
is computed memory-efficiently by remembering only s(x) = sigmoid(x - 1), since
double_swish'(x) = double_swish(x) * (1 - s(x)) + s(x).

ActivationBalancer: modifies backpropped derivatives so that, per channel, the
activation is positive at least a proportion of the time and its
average-absolute-value stays within [min_abs, max_abs], scaling negative and
positive derivative values by up to (1 +/- max_factor). It is applied randomly
with a probability that decays from 0.5 toward min_prob over training.

BalancedDoubleSwish: an ActivationBalancer followed by a DoubleSwish.
"""
import random
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class DoubleSwishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        requires_grad = x.requires_grad
        x_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)

        s = torch.sigmoid(x - 1.0)
        y = x * s

        if requires_grad:
            deriv = y * (1 - s) + s
            floor = -0.043637
            ceil = 1.2
            d_scaled = (deriv - floor) * (255.0 / (ceil - floor)) + torch.rand_like(deriv)
            if __name__ == "__main__":
                assert d_scaled.min() >= 0.0
                assert d_scaled.max() < 256.0
            d_int = d_scaled.to(torch.uint8)
            ctx.save_for_backward(d_int)
        if x.dtype == torch.float16 or torch.is_autocast_enabled():
            y = y.to(torch.float16)
        return y

    @staticmethod
    def backward(ctx, y_grad: Tensor) -> Tensor:
        (d,) = ctx.saved_tensors
        floor = -0.043637
        ceil = 1.2
        d = d * ((ceil - floor) / 255.0) + floor
        return y_grad * d


class DoubleSwish(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return x * torch.sigmoid(x - 1.0)
        return DoubleSwishFunction.apply(x)


class ActivationBalancerFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        scale_factor: Tensor,
        sign_factor: Optional[Tensor],
        channel_dim: int,
    ) -> Tensor:
        if channel_dim < 0:
            channel_dim += x.ndim
        ctx.channel_dim = channel_dim
        xgt0 = x > 0
        if sign_factor is None:
            ctx.save_for_backward(xgt0, scale_factor)
        else:
            ctx.save_for_backward(xgt0, scale_factor, sign_factor)
        return x

    @staticmethod
    def backward(ctx, x_grad: Tensor) -> Tuple[Tensor, None, None, None]:
        if len(ctx.saved_tensors) == 3:
            xgt0, scale_factor, sign_factor = ctx.saved_tensors
            for _ in range(ctx.channel_dim, x_grad.ndim - 1):
                scale_factor = scale_factor.unsqueeze(-1)
                sign_factor = sign_factor.unsqueeze(-1)
            factor = sign_factor + scale_factor * (xgt0.to(x_grad.dtype) - 0.5)
        else:
            xgt0, scale_factor = ctx.saved_tensors
            for _ in range(ctx.channel_dim, x_grad.ndim - 1):
                scale_factor = scale_factor.unsqueeze(-1)
            factor = scale_factor * (xgt0.to(x_grad.dtype) - 0.5)
        neg_delta_grad = x_grad.abs() * factor
        return (
            x_grad - neg_delta_grad,
            None,
            None,
            None,
        )


def _compute_scale_factor(
    x: Tensor,
    channel_dim: int,
    min_abs: float,
    max_abs: float,
    gain_factor: float,
    max_factor: float,
) -> Tensor:
    if channel_dim < 0:
        channel_dim += x.ndim
    sum_dims = [d for d in range(x.ndim) if d != channel_dim]
    x_abs_mean = torch.mean(x.abs(), dim=sum_dims).to(torch.float32)

    if min_abs == 0.0:
        below_threshold = 0.0
    else:
        below_threshold = ((min_abs - x_abs_mean) * (gain_factor / min_abs)).clamp(min=0, max=max_factor)

    above_threshold = ((x_abs_mean - max_abs) * (gain_factor / max_abs)).clamp(min=0, max=max_factor)

    return below_threshold - above_threshold


def _compute_sign_factor(
    x: Tensor,
    channel_dim: int,
    min_positive: float,
    max_positive: float,
    gain_factor: float,
    max_factor: float,
) -> Tensor:
    if channel_dim < 0:
        channel_dim += x.ndim
    sum_dims = [d for d in range(x.ndim) if d != channel_dim]
    proportion_positive = torch.mean((x > 0).to(torch.float32), dim=sum_dims)
    if min_positive == 0.0:
        factor1 = 0.0
    else:
        factor1 = ((min_positive - proportion_positive) * (gain_factor / min_positive)).clamp_(min=0, max=max_factor)

    if max_positive == 1.0:
        factor2 = 0.0
    else:
        factor2 = ((proportion_positive - max_positive) * (gain_factor / (1.0 - max_positive))).clamp_(
            min=0, max=max_factor
        )
    sign_factor = factor1 - factor2
    assert not isinstance(sign_factor, float)
    return sign_factor


class ActivationBalancer(torch.nn.Module):
    def __init__(
        self,
        num_channels: int,
        channel_dim: int,
        min_positive: float = 0.05,
        max_positive: float = 0.95,
        max_factor: float = 0.04,
        sign_gain_factor: float = 0.01,
        scale_gain_factor: float = 0.02,
        min_abs: float = 0.2,
        max_abs: float = 100.0,
        min_prob: float = 0.1,
    ):
        super(ActivationBalancer, self).__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.min_positive = min_positive
        self.max_positive = max_positive
        self.max_factor = max_factor
        self.min_abs = min_abs
        self.max_abs = max_abs
        self.min_prob = min_prob
        self.sign_gain_factor = sign_gain_factor
        self.scale_gain_factor = scale_gain_factor

        self.cpu_count = 0
        self.register_buffer("count", torch.tensor(0, dtype=torch.int64))

    def forward(self, x: Tensor) -> Tensor:
        if torch.jit.is_scripting() or not x.requires_grad or torch.jit.is_tracing():
            return _no_op(x)

        count = self.cpu_count
        self.cpu_count += 1

        if random.random() < 0.01:
            self.cpu_count = max(self.cpu_count, self.count.item())
            self.count.fill_(self.cpu_count)

        prob = max(self.min_prob, 0.5 ** (1 + (count / 4000.0)))

        if random.random() < prob:
            sign_gain_factor = 0.5
            if self.min_positive != 0.0 or self.max_positive != 1.0:
                sign_factor = _compute_sign_factor(
                    x,
                    self.channel_dim,
                    self.min_positive,
                    self.max_positive,
                    gain_factor=self.sign_gain_factor / prob,
                    max_factor=self.max_factor,
                )
            else:
                sign_factor = None

            scale_factor = _compute_scale_factor(
                x.detach(),
                self.channel_dim,
                min_abs=self.min_abs,
                max_abs=self.max_abs,
                gain_factor=self.scale_gain_factor / prob,
                max_factor=self.max_factor,
            )
            return ActivationBalancerFunction.apply(
                x,
                scale_factor,
                sign_factor,
                self.channel_dim,
            )
        else:
            return _no_op(x)


def BalancedDoubleSwish(d_model, channel_dim=-1, max_abs=10.0, min_prob=0.25) -> nn.Sequential:
    balancer = ActivationBalancer(d_model, channel_dim=channel_dim, max_abs=max_abs, min_prob=min_prob)
    return nn.Sequential(
        balancer,
        DoubleSwish(),
    )
