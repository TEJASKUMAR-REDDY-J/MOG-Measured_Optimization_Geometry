"""Per-block geometry measurements (the "atlas") and the selection proxy.

Candidate set per 2-D block (PDF §5, architecture-informed):
    sign (elementwise; also the proxy geometry for adam), euclid, rows, cols,
    spectral (exact SVD; also the proxy geometry for normuon / Muon),
    spectral_head (attention only), pw05 (alpha = 1/2).
"""

from __future__ import annotations

import torch

from mog.geometry import Spectral
from mog.optim.blockwise import ATTN, geometry_for
from mog.selection.metrics import effective_density, effective_neurons, effective_rank, normalized_gradient_fit

PROXY_GEOMETRY = {"adam": "sign", "adam_mini": "sign", "lion": "sign", "normuon": "spectral"}
EXACT_SPECTRAL = Spectral(method="svd")
NS_SPECTRAL = Spectral(method="ns")


def candidate_names(block: dict) -> list[str]:
    names = ["sign", "euclid", "rows", "cols", "spectral"]
    if block["kind"] in ATTN:
        names.append("spectral_head")
    return names + ["pw05"]


def candidate_geometry(name: str, block: dict):
    return EXACT_SPECTRAL if name == "spectral" else geometry_for(name, block)


def static_metrics(block: dict, B: torch.Tensor) -> dict:
    """Metrics that need only the momentum B."""
    m, n = B.shape
    out = {
        "r_eff_frac": effective_rank(B) / min(m, n),
        "d_eff_frac": effective_density(B) / (m * n),
        "m_eff_frac": effective_neurons(B) / m,
    }
    exact = float(EXACT_SPECTRAL.exact_dual_norm(B))
    ns_dir = NS_SPECTRAL.lmo(B)
    ex_dir = EXACT_SPECTRAL.lmo(B)
    out["ns_dual_rel_err"] = float((B * ns_dir).sum()) / exact - 1
    out["ns_cos"] = float((ns_dir * ex_dir).sum() / (ns_dir.norm() * ex_dir.norm()))
    for c in candidate_names(block):
        out[f"fit_{c}"] = normalized_gradient_fit(candidate_geometry(c, block), B)
    return out


def secants(block: dict, dg: torch.Tensor, delta: torch.Tensor) -> dict:
    """L_hat_c = ||dg||_{*,c} / ||delta||_c for every candidate c."""
    out = {}
    for c in candidate_names(block):
        g = candidate_geometry(c, block)
        out[c] = float(g.exact_dual_norm(dg) if isinstance(g, Spectral) else g.dual_norm(dg)) / float(g.norm(delta))
    return out


def scores(block: dict, B: torch.Tensor, L: dict) -> dict:
    """E_c = theta_c / L_hat_c with theta_c = ||B||_{*,c}^2 / ||B||_F^2 (PDF §5)."""
    out = {}
    for c, Lc in L.items():
        g = candidate_geometry(c, block)
        dual = g.exact_dual_norm(B) if isinstance(g, Spectral) else g.dual_norm(B)
        out[c] = float(dual**2 / B.norm() ** 2) / Lc
    return out


def participation_ratio(x: torch.Tensor) -> float:
    """(sum lambda)^2 / sum lambda^2 of the input-activation covariance, / dim."""
    x = x.reshape(-1, x.shape[-1]).double()
    x = x - x.mean(0)
    ev = torch.linalg.eigvalsh(x.T @ x / x.shape[0]).clamp_min(0)
    return float(ev.sum() ** 2 / (ev**2).sum()) / x.shape[-1]
