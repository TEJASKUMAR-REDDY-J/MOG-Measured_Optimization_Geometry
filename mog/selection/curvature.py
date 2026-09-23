"""Secant curvature estimate in a chosen norm (PDF §4.4; Gluon's "trajectory smoothness").

    L_hat_c = ||G_t - G_{t-1}||_{*,c} / ||Delta_{t-1}||_c

Two caveats recorded in THEORY.md §4:
  * On a quadratic, <D, H D> <= ||H D||_* ||D||, so L_hat_c upper-bounds the
    directional Rayleigh quotient <D,HD>/||D||^2 along the step taken -- it is
    NOT the sup-curvature L_c of Prop. 1.
  * Delta is the step actually taken, i.e. the incumbent geometry's direction.
    For non-incumbent candidates it measures curvature along the wrong direction.
Single-step values are noisy; callers should EMA-smooth.
"""

from __future__ import annotations

from torch import Tensor

from mog.geometry import Geometry


def secant_curvature(geometry: Geometry, g_prev: Tensor, g_cur: Tensor, delta: Tensor) -> float:
    return float(geometry.dual_norm(g_cur - g_prev) / geometry.norm(delta))
