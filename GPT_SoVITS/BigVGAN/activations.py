"""Snake-based periodic activation functions.

Implementation adapted from https://github.com/EdwardDixon/snake under the MIT license
(LICENSE is in incl_licenses directory).

Snake: x + 1/a * sin^2(xa), a sine-based periodic activation (Liu Ziyin, Tilman
Hartwig, Masahito Ueda, https://arxiv.org/abs/2006.08195). SnakeBeta is a modified
variant using separate trainable parameters for frequency (alpha) and magnitude (beta).
Input/output shape is (B, C, T).
"""

import torch
from torch import nn, sin, pow
from torch.nn import Parameter


class Snake(nn.Module):
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        super(Snake, self).__init__()
        self.in_features = in_features

        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        x = x + (1.0 / (alpha + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x


class SnakeBeta(nn.Module):
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        super(SnakeBeta, self).__init__()
        self.in_features = in_features

        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x
