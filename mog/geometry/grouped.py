"""Grouped spectral geometry: k-row (or k-column) blocks, per-head, per-expert.

The implementation lives in `Spectral(k=...)` (spectral.py): whole-matrix spectral is
exactly the k = m case, so one class serves both and the tests check both limits.
This module gives the grouped case its own name and the per-head constructor.
"""

from __future__ import annotations

import torch

from .spectral import Spectral


class GroupedSpectral(Spectral):
    """Spectral LMO applied independently to groups of `k` rows (axis="rows") or columns."""

    def __init__(self, k: int, axis: str = "rows", **kw):
        super().__init__(k=k, axis=axis, **kw)


class PermutedGroupedSpectral(GroupedSpectral):
    """Grouped spectral over a fixed permutation of rows (or cols): groups of the same size
    k, but with membership ignoring the architecture. The random-partition control for H4."""

    def __init__(self, k: int, perm: torch.Tensor, axis: str = "rows", **kw):
        super().__init__(k=k, axis=axis, **kw)
        self.perm, self.inv = perm, torch.argsort(perm)
        self._dim = -2 if axis == "rows" else -1
        self.name += "_perm"

    def _to_blocks(self, B):
        return super()._to_blocks(B.index_select(self._dim, self.perm))

    def _from_blocks(self, X, like):
        return super()._from_blocks(X, like).index_select(self._dim, self.inv)


def per_head(weight_shape: torch.Size, n_head: int, kind: str) -> GroupedSpectral:
    """Per-head blocks for attention projections: q/k/v group output rows, o groups input columns."""
    if kind == "o":
        return GroupedSpectral(k=weight_shape[1] // n_head, axis="cols")
    return GroupedSpectral(k=weight_shape[0] // n_head, axis="rows")
