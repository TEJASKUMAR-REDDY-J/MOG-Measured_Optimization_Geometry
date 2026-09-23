from __future__ import annotations

import math

import torch
from torch import Tensor

from .base import EPS, Geometry


class RowNorm(Geometry):
    """max over rows of the row l2 norm; lmo normalizes each row to unit l2.

    axis="rows": a row is B[..., i, :] -- for nn.Linear weight [out, in] this is
    one output neuron's incoming weights; for nn.Embedding [V, d] one token.
    axis="cols": a column is B[..., :, j] (per-input-unit / per-token for [d, V]).
    C = number of rows (resp. columns) = numel / row length.
    1-D tensors are treated as a single row.
    """

    def __init__(self, axis: str = "rows"):
        if axis not in ("rows", "cols"):
            raise ValueError(f"axis must be 'rows' or 'cols', got {axis!r}")
        self.axis = axis
        self.name = axis
        self._dim = -1 if axis == "rows" else -2

    def _row_norms(self, B: Tensor) -> Tensor:
        dim = -1 if B.ndim == 1 else self._dim
        return B.norm(dim=dim, keepdim=True)

    def lmo(self, B: Tensor) -> Tensor:
        return B / self._row_norms(B).clamp_min(EPS)

    def norm(self, D: Tensor) -> Tensor:
        return self._row_norms(D).max()

    def dual_norm(self, B: Tensor) -> Tensor:
        return self._row_norms(B).sum()

    def C(self, shape: torch.Size) -> float:
        row_len = shape[-1] if len(shape) == 1 else shape[self._dim]
        return float(math.prod(shape) // row_len)
