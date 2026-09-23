"""Gradient-fit and effective-dimension metrics (PDF §4.2-4.4).

All metrics go through the Geometry interface; nothing here knows how a
geometry computes its LMO.
"""

from __future__ import annotations

import torch
from torch import Tensor

from mog.geometry import Geometry


def gradient_fit(geometry: Geometry, B: Tensor) -> float:
    """theta_c = ||B||_{*,c}^2 / ||B||_F^2  (PDF §4.2)."""
    return float(geometry.dual_norm(B) ** 2 / B.norm() ** 2)


def normalized_gradient_fit(geometry: Geometry, B: Tensor) -> float:
    """theta_c / C_c in (0, 1]; equals 1 iff lmo_c(B) is parallel to B (Cauchy-Schwarz)."""
    return gradient_fit(geometry, B) / geometry.C(B.shape)


def effective_rank(B: Tensor) -> float:
    """r_eff = ||B||_nuc^2 / ||B||_F^2 (per matrix; leading dims summed)."""
    s = torch.linalg.svdvals(B)
    return float(s.sum() ** 2 / (s**2).sum())


def effective_density(B: Tensor) -> float:
    """d_eff = ||B||_1^2 / ||B||_F^2."""
    return float(B.abs().sum() ** 2 / B.norm() ** 2)


def effective_neurons(B: Tensor) -> float:
    """m_eff = (sum_i ||B_i,:||_2)^2 / ||B||_F^2."""
    return float(B.norm(dim=-1).sum() ** 2 / B.norm() ** 2)


def descent_efficiency(theta: float, curvature: float) -> float:
    """E_c = theta_c / L_hat_c, the selection score of PDF §5."""
    return theta / curvature
