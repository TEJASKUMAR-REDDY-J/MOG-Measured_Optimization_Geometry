from __future__ import annotations

import math

import torch
from torch import Tensor

from .base import Geometry


class Elementwise(Geometry):
    """||D||_max = max |D_ij|. lmo = sign(B) (signSGD; AdamW is a smoothed version). C = numel."""

    name = "elementwise"

    def lmo(self, B: Tensor) -> Tensor:
        return torch.sign(B)

    def norm(self, D: Tensor) -> Tensor:
        return D.abs().max()

    def dual_norm(self, B: Tensor) -> Tensor:
        return B.abs().sum()  # identical to <B, sign(B)>, cheaper

    def C(self, shape: torch.Size) -> float:
        return float(math.prod(shape))
