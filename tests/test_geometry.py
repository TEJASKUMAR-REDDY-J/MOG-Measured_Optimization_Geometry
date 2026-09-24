"""Stage 0: every geometry must satisfy the LMO/duality identities of PDF §2.1."""

import itertools
import math

import pytest
import torch

from mog.geometry import Elementwise, Euclidean, RowNorm, Spectral, newton_schulz

torch.manual_seed(0)
DT = torch.float64

EXACT = [
    Euclidean(),
    Elementwise(),
    RowNorm("rows"),
    RowNorm("cols"),
    Spectral(method="svd"),
    Spectral(k=4, method="svd"),
    Spectral(k=4, axis="cols", method="svd"),
    Spectral(alpha=0.5, method="svd"),
    Spectral(k=2, alpha=0.5, method="svd"),
]
SHAPES = [(8, 12), (12, 8), (3, 8, 12)]  # wide, tall, batched (per-expert)


@pytest.mark.parametrize("geo,shape", list(itertools.product(EXACT, SHAPES)), ids=str)
def test_lmo_is_unit_and_attains_dual(geo, shape):
    B = torch.randn(shape, dtype=DT)
    D = geo.lmo(B)
    assert D.shape == B.shape
    assert geo.norm(D).item() == pytest.approx(1.0, rel=1e-9)
    # dual norm computed independently where available
    dual = geo.exact_dual_norm(B) if isinstance(geo, Spectral) else geo.dual_norm(B)
    assert (B * D).sum().item() == pytest.approx(dual.item(), rel=1e-9)


@pytest.mark.parametrize("geo", EXACT, ids=str)
def test_lmo_beats_random_feasible_directions(geo):
    """<B, lmo(B)> >= <B, D> for any D with ||D|| <= 1 (definition of the LMO)."""
    B = torch.randn(8, 12, dtype=DT)
    best = (B * geo.lmo(B)).sum()
    for _ in range(200):
        D = torch.randn(8, 12, dtype=DT)
        D = D / geo.norm(D)
        assert (B * D).sum() <= best + 1e-9


@pytest.mark.parametrize("geo", EXACT, ids=str)
def test_C_is_sup_and_attained(geo):
    """C = sup ||D||_F^2/||D||^2: random D never exceed it, and some D attains it."""
    shape = torch.Size([8, 16])
    C = geo.C(shape)
    for _ in range(200):
        D = torch.randn(shape, dtype=DT)
        assert (D.norm() ** 2 / geo.norm(D) ** 2).item() <= C * (1 + 1e-9)
    # 8 rows of a 16x16 Sylvester-Hadamard matrix: +-1 entries, orthogonal equal-norm rows,
    # equal-norm columns orthogonal within aligned groups -> flat in every geometry, attains C.
    H = torch.tensor([[(-1) ** bin(i & j).count("1") for j in range(16)] for i in range(16)], dtype=DT)
    D = H[:8]
    assert (D.norm() ** 2 / geo.norm(D) ** 2).item() == pytest.approx(C, rel=1e-9)


def test_C_table_values():
    s = torch.Size([6, 10])
    assert Euclidean().C(s) == 1
    assert Elementwise().C(s) == 60
    assert RowNorm("rows").C(s) == 6
    assert RowNorm("cols").C(s) == 10
    assert Spectral().C(s) == 6                    # min(m, n)
    assert Spectral(k=2).C(s) == 3 * 2             # (m/k) min(k, n)
    assert Spectral(alpha=1.0, method="svd").C(s) == 1  # alpha=1 whole matrix is Euclidean


def test_special_cases_coincide():
    B = torch.randn(8, 12, dtype=DT)
    # k=1 grouped spectral == row normalization
    assert torch.allclose(Spectral(k=1, method="svd").lmo(B), RowNorm("rows").lmo(B))
    # k=m grouped == whole-matrix spectral
    assert torch.allclose(Spectral(k=8, method="svd").lmo(B), Spectral(method="svd").lmo(B))
    # alpha=1 whole matrix == Euclidean
    assert torch.allclose(Spectral(alpha=1.0, method="svd").lmo(B), Euclidean().lmo(B))
    # alpha=1 on k=1 groups == row normalization
    assert torch.allclose(Spectral(k=1, alpha=1.0, method="svd").lmo(B), RowNorm("rows").lmo(B))


def test_partial_whitening_identity():
    """U S^a V^T = (B B^T)^((a-1)/2) B (PDF §3.1), for full-row-rank B."""
    B = torch.randn(6, 10, dtype=DT)
    a = 0.5
    U, s, Vh = torch.linalg.svd(B, full_matrices=False)
    lhs = U @ torch.diag(s**a) @ Vh
    evals, evecs = torch.linalg.eigh(B @ B.T)
    rhs = evecs @ torch.diag(evals ** ((a - 1) / 2)) @ evecs.T @ B
    assert torch.allclose(lhs, rhs, atol=1e-10)


def test_newton_schulz_approximates_polar_factor():
    """Muon's quintic NS is approximate: singular values land near 1 but not at 1."""
    B = torch.randn(64, 128, dtype=torch.float32)
    X = newton_schulz(B, steps=5)
    s = torch.linalg.svdvals(X)
    assert s.min() > 0.5 and s.max() < 1.3
    U, _, Vh = torch.linalg.svd(B, full_matrices=False)
    cos = (X * (U @ Vh)).sum() / (X.norm() * (U @ Vh).norm())
    assert cos > 0.98


def test_zero_block_is_safe():
    for geo in EXACT + [Spectral()]:
        D = geo.lmo(torch.zeros(8, 12, dtype=DT))
        assert torch.isfinite(D).all()


def test_invalid_group_size_raises():
    with pytest.raises(ValueError):
        Spectral(k=5).lmo(torch.randn(8, 12))


def test_ns_cost_matches_pdf_scaling():
    """Prop. 6: NS cost ~ 4 T k m n FLOPs over m/k row blocks (plus the k^3 term)."""
    m, n, k, T = 512, 512, 64, 5
    cost = Spectral(k=k, ns_steps=T).cost(torch.Size([m, n]))
    assert cost == pytest.approx((m // k) * T * (4 * k * k * n + 2 * k**3))
    assert cost / (4 * T * k * m * n) == pytest.approx(1 + k / (2 * n))
    assert math.isclose(Spectral(k=1).cost(torch.Size([m, n])) / Spectral(k=2).cost(torch.Size([m, n])), 0.5, rel_tol=0.01)


def test_permuted_grouping_is_grouped_spectral_on_permuted_rows():
    from mog.geometry.grouped import GroupedSpectral, PermutedGroupedSpectral
    B = torch.randn(8, 12, dtype=DT)
    ident = PermutedGroupedSpectral(4, torch.arange(8), method="svd")
    assert torch.allclose(ident.lmo(B), GroupedSpectral(4, method="svd").lmo(B))
    perm = torch.randperm(8, generator=torch.Generator().manual_seed(0))
    g = PermutedGroupedSpectral(4, perm, method="svd")
    D = g.lmo(B)
    assert torch.allclose(D[perm], GroupedSpectral(4, method="svd").lmo(B[perm]))
    assert g.norm(D).item() == pytest.approx(1.0, rel=1e-9)
    assert (B * D).sum().item() == pytest.approx(g.exact_dual_norm(B).item(), rel=1e-9)
