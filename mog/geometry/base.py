"""Common interface for update geometries.

A geometry is a norm ||.|| on a parameter block. Everything the rest of the
library needs from it goes through four operations (THEORY.md §1):

    lmo(B)        argmax_{||D|| <= 1} <B, D>      (steepest-descent direction)
    norm(D)       primal norm ||D||
    dual_norm(B)  ||B||_* = <B, lmo(B)>            (Prop. 2: one LMO call)
    C(shape)      sup_D ||D||_F^2 / ||D||^2        (geometric constant, §2.1 table)

The optimizer, the oracle and the selector only ever call these; they never
look inside a geometry.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
from torch import Tensor

EPS = 1e-30  # guards 0/0 for an all-zero block; lmo(0) = 0


class Geometry(ABC):
    name: str = "geometry"

    @abstractmethod
    def lmo(self, B: Tensor) -> Tensor:
        """Unit-primal-norm direction D maximizing <B, D>. Same shape as B."""

    @abstractmethod
    def norm(self, D: Tensor) -> Tensor:
        """Primal norm of D (0-dim tensor)."""

    @abstractmethod
    def C(self, shape: torch.Size) -> float:
        """sup ||D||_F^2 / ||D||^2 over nonzero D of this shape."""

    def dual_norm(self, B: Tensor) -> Tensor:
        # Exact when lmo is exact; for Newton-Schulz this is the cheap proxy
        # the PDF proposes (Prop. 2), and its error is itself measured (Exp 0).
        return (B * self.lmo(B)).sum()

    def cost(self, shape: torch.Size) -> float:
        """Approximate FLOPs of one lmo call. Default: a few passes over the block."""
        return 3.0 * math.prod(shape)

    def __repr__(self) -> str:
        return self.name


class Euclidean(Geometry):
    """||D||_F. lmo = B / ||B||_F (normalized momentum SGD). C = 1."""

    name = "euclidean"

    def lmo(self, B: Tensor) -> Tensor:
        return B / B.norm().clamp_min(EPS)

    def norm(self, D: Tensor) -> Tensor:
        return D.norm()

    def C(self, shape: torch.Size) -> float:
        return 1.0
