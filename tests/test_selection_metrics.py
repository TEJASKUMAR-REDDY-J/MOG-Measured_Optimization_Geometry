"""Stage 0 checks on gradient-fit (PDF §4.2-4.3). Only closed-form facts are asserted here;
the Monte-Carlo columns of the §4.3 table are Exp 0's job, not a unit test's."""

import pytest
import torch

from mog.geometry import Elementwise, Euclidean, RowNorm, Spectral
from mog.selection import (
    effective_density,
    effective_neurons,
    effective_rank,
    gradient_fit,
    normalized_gradient_fit,
)

DT = torch.float64
GEOS = [Euclidean(), Elementwise(), RowNorm("rows"), RowNorm("cols"), Spectral(method="svd"), Spectral(k=4, method="svd")]


@pytest.mark.parametrize("geo", GEOS, ids=str)
def test_cauchy_schwarz_bound(geo):
    """theta_c / C_c <= 1 (PDF §4.2)."""
    for _ in range(20):
        assert normalized_gradient_fit(geo, torch.randn(8, 16, dtype=DT)) <= 1 + 1e-12


@pytest.mark.parametrize("geo", GEOS, ids=str)
def test_equality_when_lmo_parallel_to_B(geo):
    """Equality iff lmo(B) || B: feeding B = lmo(anything flat) must give exactly 1."""
    H = torch.tensor([[(-1) ** bin(i & j).count("1") for j in range(16)] for i in range(16)], dtype=DT)
    assert normalized_gradient_fit(geo, H[:8]) == pytest.approx(1.0, rel=1e-9)


def test_euclidean_fit_is_always_one():
    assert normalized_gradient_fit(Euclidean(), torch.randn(5, 7, dtype=DT)) == pytest.approx(1.0)


def test_effective_dimensions_match_special_cases():
    """theta_spec/C = r_eff/min(m,n), theta_rows/C = m_eff/m, theta_max/C = d_eff/(mn)."""
    B = torch.randn(8, 16, dtype=DT)
    assert normalized_gradient_fit(Spectral(method="svd"), B) == pytest.approx(effective_rank(B) / 8)
    assert normalized_gradient_fit(RowNorm("rows"), B) == pytest.approx(effective_neurons(B) / 8)
    assert normalized_gradient_fit(Elementwise(), B) == pytest.approx(effective_density(B) / 128)
    assert gradient_fit(Spectral(method="svd"), B) == pytest.approx(effective_rank(B))


@pytest.mark.parametrize("beta,r_eff_pdf,full_pdf", [(0.0, 512, 1.000), (0.5, 282, 0.550), (1.0, 28.3, 0.055), (1.5, 5.30, 0.010)])
def test_pdf_table_closed_form_columns(beta, r_eff_pdf, full_pdf):
    """r_eff and the full-spectral column of PDF §4.3 depend only on the spectrum, so they
    are closed-form. Tolerances = the PDF's printed precision (half a unit in the last digit,
    plus 0.001 slack for the PDF rounding 0.5508 -> 0.550)."""
    s = torch.arange(1, 513, dtype=DT) ** (-beta)
    B = torch.diag(s)
    r_eff = effective_rank(B)
    assert r_eff == pytest.approx(r_eff_pdf, rel=0.005)
    assert normalized_gradient_fit(Spectral(method="svd"), B) == pytest.approx(full_pdf, abs=0.0015)
