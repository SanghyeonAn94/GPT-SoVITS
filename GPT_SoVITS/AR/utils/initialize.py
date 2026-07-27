#!/usr/bin/env python3
"""Initialize weights of espnet2 neural network modules.

initialize(model, init) applies the given method (xavier_uniform, xavier_normal, kaiming_uniform,
kaiming_normal) to weights and zeros biases. Custom modules may implement espnet_initialization_fn.
"""

import torch
from typeguard import check_argument_types


def initialize(model: torch.nn.Module, init: str):
    assert check_argument_types()
    print("init with", init)

    for p in model.parameters():
        if p.dim() > 1:
            if init == "xavier_uniform":
                torch.nn.init.xavier_uniform_(p.data)
            elif init == "xavier_normal":
                torch.nn.init.xavier_normal_(p.data)
            elif init == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(p.data, nonlinearity="relu")
            elif init == "kaiming_normal":
                torch.nn.init.kaiming_normal_(p.data, nonlinearity="relu")
            else:
                raise ValueError("Unknown initialization: " + init)
    for name, p in model.named_parameters():
        if ".bias" in name and p.dim() == 1:
            p.data.zero_()
