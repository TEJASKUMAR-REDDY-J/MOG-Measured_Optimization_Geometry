"""Secant curvature (PDF §4.4) on quadratics f(W) = 0.5 <W, H[W]>, where G_t - G_{t-1} = H[Delta]."""

import pytest
import torch

from mog.geometry import Elementwise, Euclidean, RowNorm, Spectral
from mog.selection import secant_curvature

DT = torch.float64
GEOS = [Euclidean(), Elementwise(), RowNorm("rows"), Spectral(method="svd"), Spectral(k=4, method="svd")]


def _quadratic(diag):
    """Diagonal-in-coordinates Hessian H[W] = diag * W; returns grad(W)."""
    return lambda W: diag * W


@pytest.mark.parametrize("geo", GEOS, ids=str)
def test_isotropic_curvature_along_own_lmo_equals_h_times_C(geo):
    """Corollary §4.2: under isotropic H = h I, L_c = L_F * C_c * a_c with a_c = 1.
    The secant along the geometry's own LMO direction recovers exactly h * C_c."""
    h = 3.0
    grad = _quadratic(torch.full((8, 16), h, dtype=DT))
    H16 = torch.tensor([[(-1) ** bin(i & j).count("1") for j in range(16)] for i in range(16)], dtype=DT)
    delta = geo.lmo(H16[:8])  # a direction for which the norm ball is "flat"
    W0 = torch.randn(8, 16, dtype=DT)
    L = secant_curvature(geo, grad(W0), grad(W0 + delta), delta)
    assert L == pytest.approx(h * geo.C(delta.shape), rel=1e-9)


@pytest.mark.parametrize("geo", GEOS, ids=str)
def test_secant_upper_bounds_directional_rayleigh(geo):
    """<D, H D> <= ||H D||_* ||D||  =>  L_hat >= <D,HD>/||D||^2 (derived; THEORY.md §4)."""
    torch.manual_seed(1)
    diag = torch.rand(8, 16, dtype=DT) * 10
    grad = _quadratic(diag)
    for _ in range(20):
        W0, delta = torch.randn(8, 16, dtype=DT), torch.randn(8, 16, dtype=DT)
        L = secant_curvature(geo, grad(W0), grad(W0 + delta), delta)
        rayleigh = float((delta * diag * delta).sum() / geo.norm(delta) ** 2)
        assert L >= rayleigh * (1 - 1e-12)


@pytest.mark.parametrize("geo", GEOS, ids=str)
def test_secant_is_scale_invariant(geo):
    diag = torch.rand(8, 16, dtype=DT)
    grad = _quadratic(diag)
    W0, delta = torch.randn(8, 16, dtype=DT), torch.randn(8, 16, dtype=DT)
    a = secant_curvature(geo, grad(W0), grad(W0 + delta), delta)
    b = secant_curvature(geo, grad(W0), grad(W0 + 0.01 * delta), 0.01 * delta)
    assert a == pytest.approx(b, rel=1e-9)
