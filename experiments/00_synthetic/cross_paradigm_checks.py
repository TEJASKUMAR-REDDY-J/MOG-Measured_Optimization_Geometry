"""Exp 520 (pivot Phase 1) -- numerical checks of the cross-paradigm document's closed-form
claims, and validation of the new block-statistic estimators on planted ground truth.
Evidence class: synthetic. Tolerances are pre-registered in docs/EXPERIMENT_LOG.md.

  T1 low-SNR alignment efficiency relative to SGD: {1, 0.72, 0.64} x SNR^2 (SGD, polar, sign)
  T2 low-rank ceiling: exact-polar cosine with the true gradient saturates at sqrt(r/n)
  T3 tail amplification: noise fraction (n-r)/((n-r)+rT) = 0.070 (doc sim 0.072); eff. rank ~26 (polar) vs ~6 (SGD)
  T4 coupling law: argmax_alpha gain(D_alpha) = 1 - 4p
  T5 kappa_inf: 1 for diagonal A, n for (+-1 vector) rank one, ~2n/pi for Gaussian rank one
  T6 split-batch alignment is unbiased where the naive same-batch cosine is not
  T7 coupling_exponent recovers a planted p
  T8 power: 5 seeds/arm detect >= 2.02 SD (2.83 SD Bonferroni /6); TOST n = 69 (0.5 SD), 18 (1 SD)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from mog.selection.block_stats import coupling_exponent, kappa_inf, split_alignment

torch.set_default_dtype(torch.float64)


def polar(X):
    U, _, Vh = torch.linalg.svd(X, full_matrices=False)
    return U @ Vh


def cos(a, b):
    return (a * b).sum((-2, -1)) / (a.flatten(-2).norm(dim=-1) * b.flatten(-2).norm(dim=-1))


def lowrank(n, r, g):
    return torch.randn(n, r, generator=g) @ torch.randn(r, n, generator=g)


def eff_rank(X):  # Roy & Vetterli: exp(entropy of normalised singular values)
    s = torch.linalg.svdvals(X)
    p = s / s.sum()
    return float(torch.exp(-(p * p.clamp_min(1e-300).log()).sum()))


def t1(g, n=64, draws=4000):
    out = {}
    for snr in (0.01, 0.03):
        G = torch.randn(n, n, generator=g)
        G *= snr * n / G.norm()  # SNR = ||G||_F / sqrt(E||N||_F^2), N iid N(0,1)
        X = G + torch.randn(draws, n, n, generator=g)
        c = {"sgd": cos(X, G).mean(), "polar": cos(polar(X), G).mean(), "sign": cos(torch.sign(X), G).mean()}
        out[snr] = {k: float(v / c["sgd"]) ** 2 for k, v in c.items()}
    return out


def t2(g, n=64, r=4, draws=40):
    out = {}
    for snr in (0.1, 1.0, 10.0):
        vals = []
        for _ in range(draws):
            G = lowrank(n, r, g)
            G *= snr * n / G.norm()
            vals.append(float(cos(polar(G + torch.randn(n, n, generator=g)), G)))
        out[snr] = float(np.mean(vals))
    return out


def t3(g, n=64, r=4, T=200, snr=10.0, draws=10):
    rho, er_p, er_s = [], [], []
    for _ in range(draws):
        G = lowrank(n, r, g)
        G *= snr * n / G.norm()
        U, _, Vh = torch.linalg.svd(G)
        P, Q = U[:, :r] @ U[:, :r].T, Vh[:r].T @ Vh[:r]
        Sp = torch.zeros(n, n)
        Ss = torch.zeros(n, n)
        for _ in range(T):
            X = G + torch.randn(n, n, generator=g)
            Sp += polar(X)
            Ss += X / X.norm()
        sig = P @ Sp @ Q
        rho.append(1 - float(sig.norm() ** 2 / Sp.norm() ** 2))
        er_p.append(eff_rank(Sp))
        er_s.append(eff_rank(Ss))
    return {"noise_frac_sim": float(np.mean(rho)), "noise_frac_formula": (n - r) / ((n - r) + r * T),
            "eff_rank_polar": float(np.mean(er_p)), "eff_rank_sgd": float(np.mean(er_s))}


def t4(g):
    alphas = np.linspace(-2, 2, 4001)
    worst = 0.0
    for p in (0.0, 0.1, 0.25, 0.4, 0.5):
        s = np.sort(np.abs(np.random.default_rng(int(p * 100)).standard_normal(50)))[::-1] + 0.05
        gain = [(np.sum(s ** (1 + a))) ** 2 / (2 * np.sum(s ** (2 * a + 4 * p))) for a in alphas]
        worst = max(worst, abs(alphas[int(np.argmax(gain))] - (1 - 4 * p)))
    return {"max_abs_error": worst}


def t5(g, n=128):
    v = torch.randn(n, 1, generator=g)
    s = torch.sign(torch.randn(n, 1, generator=g))
    d = torch.rand(n, generator=g) + 0.1
    return {"diag": kappa_inf(torch.diag(d)), "sign_rank1": kappa_inf(s @ s.T),
            "gauss_rank1": kappa_inf(v @ v.T) | {"exact": float(v.abs().sum() ** 2 / (v * v).sum()),
                                                 "2n_over_pi": 2 * n / math.pi}}


def t6(g, m=64, n=128, draws=200):
    lmos = {"spectral": polar, "sign": torch.sign, "rows": lambda X: X / X.norm(dim=1, keepdim=True)}
    out = {}
    for snr in (0.3, 1.0, 3.0):
        acc = {k: {"split": [], "naive": [], "true": []} for k in lmos}
        for _ in range(draws):
            G = torch.randn(m, 8, generator=g) @ torch.randn(8, n, generator=g)
            G *= snr * math.sqrt(m * n) / G.norm()
            # half-batch gradients: noise variance 2 each, so the full-batch mean has variance 1
            G1 = G + math.sqrt(2) * torch.randn(m, n, generator=g)
            G2 = G + math.sqrt(2) * torch.randn(m, n, generator=g)
            for k, f in lmos.items():
                D = f(G1)
                acc[k]["split"].append(split_alignment(G1, G2, f))
                acc[k]["naive"].append(float(cos(D, G1)))
                acc[k]["true"].append(float(cos(D, G)))
        out[snr] = {k: {kk: float(np.nanmean(vv)) for kk, vv in a.items()} for k, a in acc.items()}
    return out


def t7(g, m=48, n=96):
    out = {}
    for p in (0.0, 0.25, 0.5):
        B = torch.randn(m, n, generator=g) * torch.linspace(1, 0.05, n)
        U, S, _ = torch.linalg.svd(B, full_matrices=False)
        Gamma = U @ torch.diag(S ** (2 * p)) @ U.T + 1e-3 * torch.eye(m) * float(S[0] ** (2 * p))
        out[p] = coupling_exponent(B, Gamma)
    return out


def t8():
    def mde(n, alpha):
        df = 2 * n - 2
        tc = stats.t.ppf(1 - alpha / 2, df)
        lo, hi = 0.0, 10.0
        for _ in range(60):
            d = (lo + hi) / 2
            nc = d * math.sqrt(n / 2)
            power = 1 - stats.nct.cdf(tc, df, nc) + stats.nct.cdf(-tc, df, nc)
            lo, hi = (d, hi) if power < 0.8 else (lo, d)
        return hi

    def tost_n(margin, alpha=0.05, power=0.8):  # normal approximation, true difference 0
        return math.ceil(2 * ((stats.norm.ppf(1 - alpha) + stats.norm.ppf(1 - (1 - power) / 2)) / margin) ** 2)

    return {"mde_5seeds": mde(5, 0.05), "mde_5seeds_bonf6": mde(5, 0.05 / 6),
            "tost_n_0.5sd": tost_n(0.5), "tost_n_1sd": tost_n(1.0)}


def verdicts(r):
    v = {}
    lo = r["T1"][0.01]
    v["T1"] = abs(lo["polar"] - 0.72) / 0.72 < 0.05 and abs(lo["sign"] - 2 / math.pi) / (2 / math.pi) < 0.05
    v["T2"] = abs(r["T2"][10.0] - 0.25) < 0.02
    t3 = r["T3"]
    v["T3"] = abs(t3["noise_frac_sim"] - 0.070) < 0.01 and abs(t3["eff_rank_polar"] - 26) <= 3 and abs(t3["eff_rank_sgd"] - 6) <= 3
    v["T4"] = r["T4"]["max_abs_error"] < 0.02
    t5 = r["T5"]
    v["T5"] = (abs(t5["diag"]["upper"] - 1) < 1e-3 and abs(t5["sign_rank1"]["lower"] - 128) < 1e-3
               and abs(t5["gauss_rank1"]["upper"] - t5["gauss_rank1"]["exact"]) / t5["gauss_rank1"]["exact"] < 1e-3
               and abs(t5["gauss_rank1"]["exact"] / t5["gauss_rank1"]["2n_over_pi"] - 1) < 0.1)
    v["T6"] = all(abs(x["split"] - x["true"]) < 0.02 for s in r["T6"].values() for x in s.values())
    v["T7"] = all(abs(x["p"] - p) < 0.02 for p, x in r["T7"].items())
    t8 = r["T8"]
    v["T8"] = (abs(t8["mde_5seeds"] - 2.02) < 0.05 and abs(t8["mde_5seeds_bonf6"] - 2.83) < 0.05
               and abs(t8["tost_n_0.5sd"] - 69) <= 1 and abs(t8["tost_n_1sd"] - 18) <= 1)
    return {k: "PASS" if ok else "FAIL" for k, ok in v.items()}


def main(cfg: dict, run_dir: Path) -> dict:
    torch.set_num_threads(cfg.get("threads", 1))
    g = torch.Generator().manual_seed(cfg["seed"])
    r = {}
    for name, fn in [("T1", t1), ("T2", t2), ("T3", t3), ("T4", t4), ("T5", t5), ("T6", t6), ("T7", t7)]:
        r[name] = fn(g)
        print(name, r[name], flush=True)
    r["T8"] = t8()
    print("T8", r["T8"])
    r["verdicts"] = verdicts(r)
    print(r["verdicts"])
    return {k: ({str(kk): vv for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in r.items()}
