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


def per_head(weight_shape: torch.Size, n_head: int, kind: str) -> GroupedSpectral:
    """Per-head blocks for attention projections: q/k/v group output rows, o groups input columns."""
    if kind == "o":
        return GroupedSpectral(k=weight_shape[1] // n_head, axis="cols")
    return GroupedSpectral(k=weight_shape[0] // n_head, axis="rows")
