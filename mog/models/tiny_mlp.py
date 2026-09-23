"""Tiny MLP for Stage-2 checks outside the transformer (and the cross-domain spin-off)."""

import torch.nn as nn


class TinyMLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, depth: int = 2):
        super().__init__()
        dims = [d_in] + [d_hidden] * depth
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.GELU()]
        self.net = nn.Sequential(*layers, nn.Linear(dims[-1], d_out))

    def forward(self, x):
        return self.net(x)
