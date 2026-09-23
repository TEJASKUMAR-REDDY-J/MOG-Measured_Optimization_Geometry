"""Exp 0 -- independent reproduction of the PDF's synthetic geometry analysis.

EVIDENCE CLASS: synthetic / closed-form. Nothing here is a training measurement.

Reproduces (conceptually) PDF §4.3 table, Fig. 1b, 1c, 1d, 2a, and adds one
check the PDF does not report: how well the Newton-Schulz output used by Muon
recovers the exact nuclear norm and U V^T direction (Prop. 2 assumes it does).

Momentum model (PDF §4.3): B = U diag(sigma) V^T, U/V Haar-random, sigma_i = i^-beta.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from mog.geometry import Elementwise, Euclidean, RowNorm, Spectral
from mog.selection import effective_rank, normalized_gradient_fit

DTYPES = {"float32": torch.float32, "float64": torch.float64, "bfloat16": torch.bfloat16}


def haar(n: int, gen: torch.Generator, dtype) -> torch.Tensor:
    Q, R = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=dtype))
    return Q * torch.sign(torch.diagonal(R))


def power_law_matrix(m: int, n: int, beta: float, gen: torch.Generator, dtype) -> torch.Tensor:
    r = min(m, n)
    U, V = haar(m, gen, dtype)[:, :r], haar(n, gen, dtype)[:, :r]
    sigma = torch.arange(1, r + 1, dtype=dtype) ** (-beta)
    return (U * sigma) @ V.T


def _stats(xs):
    a = np.asarray(xs, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0, "n": int(a.size)}


def main(cfg: dict, run_dir: Path) -> dict:
    m, n = cfg["m"], cfg["n"]
    dtype = DTYPES[cfg["dtype"]]
    seeds = [cfg["seed"] + i for i in range(cfg["n_seeds"])]
    betas = sorted(set(cfg["betas_sweep"]) | set(cfg["betas_table"]) | set(cfg["betas_grouped"]))

    base = {
        "spectral_full_svd": Spectral(method="svd"),
        "rows": RowNorm("rows"),
        "cols": RowNorm("cols"),
        "elementwise": Elementwise(),
        "euclidean": Euclidean(),
    }
    ns = {dt: Spectral(method="ns", ns_steps=cfg["ns_steps"], ns_dtype=DTYPES[dt]) for dt in cfg["ns_dtypes"]}

    rows = []
    for seed in seeds:
        gen = torch.Generator().manual_seed(seed)
        for beta in betas:
            B = power_law_matrix(m, n, beta, gen, dtype)
            r_eff = effective_rank(B)
            common = {"seed": seed, "beta": beta, "r_eff": r_eff}
            for name, g in base.items():
                rows.append(common | {"geometry": name, "k": "", "method": "exact",
                                      "theta_over_C": normalized_gradient_fit(g, B)})
            exact_dir = base["spectral_full_svd"].lmo(B)
            exact_dual = float(base["spectral_full_svd"].exact_dual_norm(B))
            for dt, g in ns.items():
                d = g.lmo(B)
                rows.append(common | {
                    "geometry": "spectral_full_ns", "k": "", "method": f"ns_{dt}",
                    "theta_over_C": normalized_gradient_fit(g, B),
                    "ns_dual_rel_err": float((B * d).sum()) / exact_dual - 1.0,
                    "ns_cos_to_exact": float((d * exact_dir).sum() / (d.norm() * exact_dir.norm())),
                })
            if beta in cfg["betas_grouped"]:
                for k in cfg["group_sizes"]:
                    gk = Spectral(k=k, method="svd")
                    rows.append(common | {"geometry": "spectral_grouped", "k": k, "method": "exact",
                                          "theta_over_C": normalized_gradient_fit(gk, B)})
        print(f"seed {seed} done")

    fields = ["seed", "beta", "geometry", "k", "method", "theta_over_C", "r_eff", "ns_dual_rel_err", "ns_cos_to_exact"]
    with open(run_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    def agg(geometry, beta, method="exact", k="", key="theta_over_C"):
        return _stats([r[key] for r in rows if r["geometry"] == geometry and r["beta"] == beta
                       and r["method"] == method and r["k"] == k])

    # --- PDF §4.3 table: reproduction vs reported values -------------------
    ref = cfg.get("pdf_reference_table", {})
    table = []
    for beta in cfg["betas_table"]:
        row = {"beta": beta, "r_eff": agg("rows", beta, key="r_eff")}
        for geo in ("spectral_full_svd", "rows", "elementwise"):
            row[geo] = agg(geo, beta)
        pdf = ref.get(str(beta)) or ref.get(beta)
        if pdf:
            row["pdf"] = pdf
            row["abs_diff"] = {
                "r_eff": abs(row["r_eff"]["mean"] - pdf["r_eff"]),
                "spectral_full_svd": abs(row["spectral_full_svd"]["mean"] - pdf["full_spectral"]),
                "rows": abs(row["rows"]["mean"] - pdf["rows"]),
                "elementwise": abs(row["elementwise"]["mean"] - pdf["elementwise"]),
            }
        table.append(row)

    grouped = {str(b): {str(k): agg("spectral_grouped", b, k=k)["mean"] for k in cfg["group_sizes"]}
               for b in cfg["betas_grouped"]}
    ns_fidelity = {dt: {str(b): {"dual_rel_err": agg("spectral_full_ns", b, f"ns_{dt}", key="ns_dual_rel_err"),
                                 "cos_to_exact": agg("spectral_full_ns", b, f"ns_{dt}", key="ns_cos_to_exact")}
                        for b in betas} for dt in cfg["ns_dtypes"]}

    # --- closed-form panels (no randomness) ------------------------------
    oc = cfg["overhead"]
    overhead = []
    for k in oc["group_sizes"]:
        cost = Spectral(k=k, ns_steps=oc["ns_steps"]).cost(torch.Size([oc["m"], oc["n"]]))
        for bt in oc["tokens_per_step"]:
            overhead.append({"k": k, "tokens_per_step": bt,
                             "pdf_formula_pct": 100 * (2 * oc["ns_steps"] / 3) * k / bt,
                             "interface_cost_pct": 100 * cost / (6 * oc["m"] * oc["n"] * bt)})
    sc = cfg["shrinkage"]
    shrink = [{"snr": nu, "k": k,
               "lambda": sc["tau"] ** 2 / (sc["tau"] ** 2 + 1.0 / (k * sc["fan_in"] * nu**2))}
              for nu in sc["snr"] for k in sc["group_sizes"]]

    _plots(cfg, rows, betas, grouped, overhead, shrink, run_dir)
    return {"table_4_3": table, "grouped_fig1c_mean": grouped, "ns_fidelity": ns_fidelity,
            "overhead_fig1d": overhead, "shrinkage_fig2a": shrink, "seeds": seeds}


def _plots(cfg, rows, betas, grouped, overhead, shrink, run_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = run_dir / "figures"
    fig_dir.mkdir()
    tag = "SYNTHETIC / closed-form - not a training measurement"

    def mean_by_beta(geometry, method="exact", key="theta_over_C"):
        return [np.mean([r[key] for r in rows if r["geometry"] == geometry and r["beta"] == b
                         and r["method"] == method and r["k"] == ""]) for b in betas]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    r = min(cfg["m"], cfg["n"])
    for b in cfg["betas_grouped"]:
        ax[0].loglog(np.arange(1, r + 1), np.arange(1, r + 1) ** (-b), label=f"beta={b}")
    ax[0].set(xlabel="i", ylabel="sigma_i", title="Synthetic spectra sigma_i = i^-beta")
    ax[0].legend()
    for geo, lab in [("spectral_full_svd", "full spectral (exact)"), ("rows", "rows"),
                     ("cols", "cols"), ("elementwise", "elementwise")]:
        ax[1].semilogy(betas, mean_by_beta(geo), "o-", label=lab)
    for dt in cfg["ns_dtypes"]:
        ax[1].semilogy(betas, mean_by_beta("spectral_full_ns", f"ns_{dt}"), "x--", label=f"full spectral (NS {dt})")
    ax[1].set(xlabel="beta", ylabel="theta_c / C_c", title="Fig 1b analogue")
    ax[1].legend(fontsize=7)
    fig.suptitle(tag, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1b_gradient_fit_vs_beta.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    for b, d in grouped.items():
        ks = [int(k) for k in d]
        ax.loglog(ks, list(d.values()), "o-", label=f"beta={b}")
    ax.set(xlabel="row-group size k", ylabel="theta_k / C_k", title="Fig 1c analogue")
    ax.legend()
    fig.suptitle(tag, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1c_gradient_fit_vs_group_size.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    for dt in cfg["ns_dtypes"]:
        ax.plot(betas, mean_by_beta("spectral_full_ns", f"ns_{dt}", "ns_dual_rel_err"), "o-", label=f"NS {dt}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel="beta", ylabel="<B, NS(B)> / ||B||_nuc - 1", title="NS nuclear-norm proxy error (Prop. 2)")
    ax.legend()
    fig.suptitle(tag, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "ns_dual_norm_fidelity.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for bt in cfg["overhead"]["tokens_per_step"]:
        o = [x for x in overhead if x["tokens_per_step"] == bt]
        ax[0].loglog([x["k"] for x in o], [x["pdf_formula_pct"] for x in o], "o-", label=f"{bt} tok/step (PDF formula)")
        ax[0].loglog([x["k"] for x in o], [x["interface_cost_pct"] for x in o], "x--", label=f"{bt} tok/step (incl. k^3 term)")
    ax[0].set(xlabel="k", ylabel="NS FLOPs, % of layer train FLOPs", title="Fig 1d analogue")
    ax[0].legend(fontsize=7)
    for nu in cfg["shrinkage"]["snr"]:
        s = [x for x in shrink if x["snr"] == nu]
        ax[1].semilogx([x["k"] for x in s], [x["lambda"] for x in s], label=f"nu={nu}")
    ax[1].axhline(0.5, ls="--", color="gray")
    ax[1].set(xlabel="group size k", ylabel="lambda_g", title="Fig 2a analogue (model-derived)")
    ax[1].legend()
    fig.suptitle(tag, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1d_fig2a_closed_form.png", dpi=120)
    plt.close(fig)
