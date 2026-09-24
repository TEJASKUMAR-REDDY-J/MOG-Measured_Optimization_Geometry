"""Block statistics of the cross-paradigm proposal (docs/pivot/CROSS_PARADIGM_ASSESSMENT.md).

    split_alignment    cosine of lmo(G1) with the true gradient, estimated without noise bias
                       from two independent half-batch gradients G1, G2
    ranks              nuclear rank (||B||_*/||B||_F)^2, stable rank ||B||_F^2/||B||_2^2,
                       elementwise rank (||B||_1/||B||_F)^2
    coupling_exponent  p in Gamma ~ U S^(2p) U^T: slope of log(u_i^T Gamma u_i) on log sigma_i^2,
                       plus the off-diagonal mass of U^T Gamma U (validity of the shared-frame assumption)
    kappa_inf          max_{s in {+-1}^n} s^T A s / tr A: SDP upper bound (Burer-Monteiro) and
                       a sign-rounded lower bound (the SDP is pi/2-accurate, Nesterov 1998)
"""

from __future__ import annotations

import math

import torch


def split_alignment(G1: torch.Tensor, G2: torch.Tensor, lmo) -> float:
    """<lmo(G1), G2> / (||lmo(G1)||_F * sqrt(<G1, G2>)).

    G2's noise is independent of G1, so the numerator is unbiased for <lmo(G1), G>; <G1, G2> is
    unbiased for ||G||_F^2. The plain cosine with G1 is biased up by G1's own noise."""
    D = lmo(G1)
    sig2 = float((G1 * G2).sum())
    if sig2 <= 0:
        return float("nan")  # no detectable signal
    return float((D * G2).sum()) / (float(D.norm()) * math.sqrt(sig2))


def ranks(B: torch.Tensor) -> dict:
    S = torch.linalg.svdvals(B)
    F2 = float((S * S).sum())
    return {"nuclear_rank": float(S.sum()) ** 2 / F2, "stable_rank": F2 / float(S[0]) ** 2,
            "elementwise_rank": float(B.abs().sum()) ** 2 / F2}


def coupling_exponent(B: torch.Tensor, Gamma: torch.Tensor, k: int | None = None) -> dict:
    """Gamma is the output-side curvature factor (m x m) for B (m x n). Fit on the top-k directions."""
    U, S, _ = torch.linalg.svd(B, full_matrices=False)
    k = k or int((S > 1e-6 * S[0]).sum())
    M = U[:, :k].T @ Gamma @ U[:, :k]
    g = torch.diagonal(M).clamp_min(1e-30).log()
    x = (S[:k] ** 2).log()
    xc = x - x.mean()
    p = float((xc * (g - g.mean())).sum() / (xc * xc).sum())
    off = float((M - torch.diag(torch.diagonal(M))).norm() / M.norm())
    return {"p": p, "offdiag_frac": off, "k": k}


def kappa_inf(A: torch.Tensor, rank: int | None = None, iters: int = 500, seed: int = 0) -> dict:
    """Bounds on max_s s^T A s / tr A for PSD A (n x n).

    Burer-Monteiro: V (n x r) with unit rows, V <- rownorm(A V) (monotone for PSD A, the
    'mixing' fixed point of max tr(A V V^T) s.t. diag(V V^T) = 1). With r > sqrt(2n) its local
    optima are global, so tr(A V V^T) is the SDP value (an upper bound on the integer max).
    Lower bound: best of 64 random-hyperplane roundings of V, plus the sign of the top eigenvector."""
    n = A.shape[0]
    r = rank or int(math.ceil(math.sqrt(2 * n))) + 1
    g = torch.Generator().manual_seed(seed)
    V = torch.randn(n, r, generator=g, dtype=A.dtype)
    V = V / V.norm(dim=1, keepdim=True)
    for _ in range(iters):
        V = A @ V
        V = V / V.norm(dim=1, keepdim=True).clamp_min(1e-30)
    sdp = float(torch.trace(V.T @ A @ V))
    cands = torch.sign(V @ torch.randn(r, 64, generator=g, dtype=A.dtype))
    cands = torch.cat([cands, torch.sign(torch.linalg.eigh(A)[1][:, -1:])], 1)
    cands[cands == 0] = 1
    lower = float(torch.einsum("ik,ij,jk->k", cands, A, cands).max())
    tr = float(torch.trace(A))
    return {"upper": sdp / tr, "lower": lower / tr}
