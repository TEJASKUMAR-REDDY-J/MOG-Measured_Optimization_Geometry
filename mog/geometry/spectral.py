"""Spectral geometries: whole-matrix (Muon) and grouped (k-row / per-head / per-expert).

One class covers the whole (G, alpha) spectral family of PDF §3.1, because
whole-matrix spectral is exactly the k = m case of grouped spectral:

    blocks  B_g = U_g S_g V_g^T   (k rows each, or k cols, or whole matrix)
    lmo_g   = U_g S_g^alpha V_g^T / ||s_g||_{1+alpha}^alpha
    norm    = max_g ||D_g||_{S_p},  p = 1 + 1/alpha   (alpha=0 -> spectral norm)
    dual    = sum_g ||B_g||_{S_{1+alpha}}             (alpha=0 -> nuclear norm)
    C       = n_blocks * r^((1-alpha)/(1+alpha)),  r = min(block dims)

Leading dims of a tensor with ndim > 2 are independent blocks (per-expert).
alpha=0 can use Newton-Schulz (the Muon polynomial) or an exact SVD; alpha>0
is SVD only for now (the polynomial route for fractional alpha is unimplemented).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .base import EPS, Geometry

# Quintic Newton-Schulz coefficients from K. Jordan's Muon reference implementation.
NS_COEFFS = (3.4445, -4.7750, 2.0315)


def newton_schulz(G: Tensor, steps: int = 5, dtype: torch.dtype = torch.float32) -> Tensor:
    """Approximate U V^T of G (batched over leading dims), as in Muon.

    Note: these coefficients do NOT converge to exact U V^T; singular values of
    the output land roughly in [0.7, 1.2]. Directional fidelity is measured in Exp 0.
    """
    a, b, c = NS_COEFFS
    X = G.to(dtype)
    tall = G.size(-2) > G.size(-1)
    if tall:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        Bm = b * A + c * A @ A
        X = a * X + Bm @ X
    if tall:
        X = X.mT
    return X


class Spectral(Geometry):
    def __init__(
        self,
        k: int | None = None,
        axis: str = "rows",
        alpha: float = 0.0,
        method: str = "ns",
        ns_steps: int = 5,
        ns_dtype: torch.dtype = torch.float32,
    ):
        if axis not in ("rows", "cols"):
            raise ValueError(f"axis must be 'rows' or 'cols', got {axis!r}")
        if method not in ("ns", "svd"):
            raise ValueError(f"method must be 'ns' or 'svd', got {method!r}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if alpha > 0 and method == "ns":
            raise NotImplementedError("alpha > 0 requires method='svd' (polynomial route not implemented)")
        self.k, self.axis, self.alpha = k, axis, alpha
        self.method, self.ns_steps, self.ns_dtype = method, ns_steps, ns_dtype
        grp = "full" if k is None else f"k{k}{'' if axis == 'rows' else 'c'}"
        self.name = f"spectral_{grp}" + (f"_a{alpha:g}" if alpha else "") + f"_{method}"

    # --- block reshaping -------------------------------------------------
    def _to_blocks(self, B: Tensor) -> Tensor:
        """[..., m, n] -> [..., G, k, n] (rows) with blocks as a batch dim."""
        if B.ndim < 2:
            raise ValueError("spectral geometry needs a >=2-D tensor")
        X = B if self.axis == "rows" else B.mT
        if self.k is None:
            return X.unsqueeze(-3)
        m = X.size(-2)
        if m % self.k:
            raise ValueError(f"group size {self.k} does not divide {m}")
        return X.reshape(*X.shape[:-2], m // self.k, self.k, X.size(-1))

    def _from_blocks(self, X: Tensor, like: Tensor) -> Tensor:
        shape = like.shape if self.axis == "rows" else like.mT.shape
        X = X.reshape(shape)
        return X if self.axis == "rows" else X.mT

    def _block_dims(self, shape: torch.Size) -> tuple[int, int, int]:
        """(n_blocks, rows_per_block, cols_per_block)."""
        m, n = (shape[-2], shape[-1]) if self.axis == "rows" else (shape[-1], shape[-2])
        k = m if self.k is None else self.k
        lead = math.prod(shape[:-2])
        return lead * (m // k), k, n

    # --- Geometry interface ---------------------------------------------
    def lmo(self, B: Tensor) -> Tensor:
        X = self._to_blocks(B)
        if self.method == "ns":
            D = newton_schulz(X, self.ns_steps, self.ns_dtype).to(B.dtype)
        else:
            U, s, Vh = torch.linalg.svd(X, full_matrices=False)
            tol = s.amax(-1, keepdim=True) * max(X.shape[-2:]) * torch.finfo(s.dtype).eps
            keep = s > tol  # drop numerically-zero directions (keeps lmo minimal)
            if self.alpha == 0:
                w = keep.to(s.dtype)
            else:
                q = 1.0 + self.alpha
                sq = (s**q).sum(-1, keepdim=True) ** (1.0 / q)
                w = torch.where(keep, s**self.alpha, 0.0) / sq.clamp_min(EPS) ** self.alpha
            D = (U * w.unsqueeze(-2)) @ Vh
        return self._from_blocks(D, B)

    def norm(self, D: Tensor) -> Tensor:
        s = torch.linalg.svdvals(self._to_blocks(D))
        if self.alpha == 0:
            return s.amax()
        p = 1.0 + 1.0 / self.alpha
        return ((s**p).sum(-1) ** (1.0 / p)).amax()

    def exact_dual_norm(self, B: Tensor) -> Tensor:
        """sum_g ||B_g||_{S_{1+alpha}} via SVD, regardless of self.method."""
        s = torch.linalg.svdvals(self._to_blocks(B))
        q = 1.0 + self.alpha
        return ((s**q).sum(-1) ** (1.0 / q)).sum()

    def C(self, shape: torch.Size) -> float:
        G, k, n = self._block_dims(shape)
        r = min(k, n)
        return G * r ** ((1.0 - self.alpha) / (1.0 + self.alpha))

    def cost(self, shape: torch.Size) -> float:
        """Newton-Schulz FLOPs: per iteration per block ~ 4 r^2 c + 2 r^3 (PDF Prop. 6 in MACs: 2k^2n + k^3)."""
        G, k, n = self._block_dims(shape)
        r, c = min(k, n), max(k, n)
        return G * self.ns_steps * (4.0 * r * r * c + 2.0 * r**3)
