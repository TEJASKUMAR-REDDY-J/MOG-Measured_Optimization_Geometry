"""Exp 640 -- proposal H1 (sufficiency): do block statistics predict the per-block geometry
benefit across paradigms? Analysis only; reads oracle (block units) and block-statistics runs.
Written and committed before any Phase 2 data existed (docs/EXPERIMENT_LOG.md, "Pivot Phase 2").

Label per (paradigm, source, checkpoint, block), dimensionless:
    y = rel_gain(spectral) - rel_gain(adam_rms)   (shape effect at matched per-block scale)
rel_gain = gain / incumbent's own progress over the same h steps (oracle_tasks.py), averaged over
the two data streams. Secondary labels: spectral - sign, spectral - adam.

Gate (label reliability): Spearman of y between streams 901 and 902, pooled over paradigms.
    < 0.3 -> labels are noise; H1 NOT TESTABLE (reported, no fit).
Test: ridge regression on standardized statistics, leave-one-paradigm-out.
    SUPPORTED   median held-out Spearman >= 0.5 and paradigm indicators add nothing (F-test p >= 0.05)
    FALSIFIED   median held-out Spearman <= 0.3, or indicators significant (p < 0.05)
    otherwise INCONCLUSIVE.
Theory check (proposal Sec. 4.4): crit = kappa_inf * r_eff / n > 2/pi predicts spectral beats sign.

Config: paradigms {name: {oracle: [run specs], stats: [run specs]}}, ridge_alpha.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from scipy import stats

from mog.selection.oracle import spearman
from mog.utils.reproducibility import REPO_ROOT

FEATURES = ["log_snr", "nuc_rank_m", "stable_rank_m", "elem_rank_m", "nuc_rank_g", "align_spectral",
            "d_align_sign", "d_align_rows", "log_crit", "p_out", "off_out", "p_in", "off_in", "tail", "boundary", "src_adamw"]
LABELS = {"shape": ("spectral", "adam_rms"), "vs_sign": ("spectral", "sign"), "vs_adam": ("spectral", "adam")}


def _run(spec):
    exp = spec.split(":", 1)[1] if spec.startswith("latest:") else spec
    return sorted((REPO_ROOT / "results" / "raw" / exp).iterdir())[-1]


def _read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else math.nan
    except (TypeError, ValueError):
        return math.nan


def load(cfg):
    """-> list of dicts: paradigm, source, ckpt, block, features, labels per stream."""
    data = []
    for par, spec in cfg["paradigms"].items():
        gains = {}
        for o in spec["oracle"]:
            for r in _read(_run(o) / "oracle_units.csv"):
                if r["unit_type"] != "block":
                    continue
                gain = _f(r.get("rel_gain")) if "rel_gain" in r else math.nan
                gains.setdefault((r["source"], int(r["ckpt"]), r["unit"]), {}).setdefault(int(r["data_seed"]), {})[r["rule"]] = gain
        st = {}
        for s in spec["stats"]:
            for r in _read(_run(s) / "block_stats.csv"):
                st[(r["source"], int(r["ckpt"]), r["block"])] = r
        for key, per_stream in gains.items():
            if key not in st:
                continue
            r = st[key]
            x = {"log_snr": math.log(max(_f(r["snr"]), 1e-8)), "boundary": float(r["role"] == "boundary"),
                 "src_adamw": float(key[0] == "adamw"), "log_crit": math.log(_f(r.get("crit"))) if _f(r.get("crit")) > 0 else math.nan,
                 "d_align_sign": _f(r["align_spectral"]) - _f(r["align_sign"]),
                 "d_align_rows": _f(r["align_spectral"]) - _f(r["align_rows"])}
            for k in ("nuc_rank_m", "stable_rank_m", "elem_rank_m", "nuc_rank_g", "align_spectral", "p_out", "off_out", "p_in", "off_in", "tail"):
                x[k] = _f(r.get(k))
            ys = {}
            for lab, (a, b) in LABELS.items():
                # the incumbent's own rule is absent from the rows: its gain is 0 by definition
                ys[lab] = {ds: g.get(a, 0.0) - g.get(b, 0.0) for ds, g in per_stream.items()}
            data.append({"paradigm": par, "source": key[0], "ckpt": key[1], "block": key[2], "x": x, "y": ys,
                         "crit": _f(r.get("crit"))})
    return data


def _matrix(rows, mu=None, sd=None):
    X = np.array([[r["x"][f] for f in FEATURES] for r in rows], float)
    if mu is None:
        mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
        sd[~np.isfinite(sd) | (sd == 0)] = 1.0
        mu[~np.isfinite(mu)] = 0.0
    X = np.where(np.isfinite(X), X, mu)
    return (X - mu) / sd, mu, sd


def ridge(X, y, alpha):
    Xa = np.c_[X, np.ones(len(X))]
    I = np.eye(Xa.shape[1])
    I[-1, -1] = 0
    return np.linalg.solve(Xa.T @ Xa + alpha * I, Xa.T @ y)


def main(cfg: dict, run_dir: Path) -> dict:
    data = load(cfg)
    out = {"n_rows": len(data), "per_paradigm_n": {p: sum(r["paradigm"] == p for r in data) for p in cfg["paradigms"]}}
    for lab in LABELS:
        rows = [r for r in data if len(r["y"][lab]) >= 2 and all(np.isfinite(list(r["y"][lab].values())))]
        s1 = [r["y"][lab][901] for r in rows]
        s2 = [r["y"][lab][902] for r in rows]
        rel = spearman(np.array(s1), np.array(s2))
        res = {"n": len(rows), "label_reliability_spearman": rel,
               "per_paradigm_reliability": {p: spearman(np.array([r["y"][lab][901] for r in rows if r["paradigm"] == p]),
                                                        np.array([r["y"][lab][902] for r in rows if r["paradigm"] == p]))
                                            for p in cfg["paradigms"]}}
        y_all = np.array([np.mean(list(r["y"][lab].values())) for r in rows])
        res["single_feature_spearman"] = {}
        X_all, _, _ = _matrix(rows)
        for j, f in enumerate(FEATURES):
            res["single_feature_spearman"][f] = spearman(X_all[:, j], y_all)
        if rel < 0.3:
            res["verdict"] = "NOT TESTABLE (label reliability < 0.3)"
        else:
            rhos = {}
            for p in cfg["paradigms"]:
                tr = [i for i, r in enumerate(rows) if r["paradigm"] != p]
                te = [i for i, r in enumerate(rows) if r["paradigm"] == p]
                if len(te) < 5:
                    continue
                Xtr, mu, sd = _matrix([rows[i] for i in tr])
                Xte, _, _ = _matrix([rows[i] for i in te], mu, sd)
                w = ridge(Xtr, y_all[tr], cfg.get("ridge_alpha", 1.0))
                rhos[p] = spearman(np.c_[Xte, np.ones(len(te))] @ w, y_all[te])
            # F-test: do paradigm indicators explain residual variance beyond the statistics?
            pars = sorted({r["paradigm"] for r in rows})
            D = np.array([[float(r["paradigm"] == p) for p in pars[1:]] for r in rows])
            X0 = np.c_[X_all, np.ones(len(rows))]
            X1 = np.c_[X0, D]
            rss = lambda X: float(np.sum((y_all - X @ np.linalg.lstsq(X, y_all, rcond=None)[0]) ** 2))
            r0, r1 = rss(X0), rss(X1)
            df1, df2 = D.shape[1], len(rows) - X1.shape[1]
            Fv = ((r0 - r1) / df1) / (r1 / df2) if df2 > 0 and r1 > 0 else math.nan
            p_ind = float(stats.f.sf(Fv, df1, df2)) if np.isfinite(Fv) else math.nan
            med = float(np.median(list(rhos.values()))) if rhos else math.nan
            verdict = ("SUPPORTED" if med >= 0.5 and p_ind >= 0.05 else
                       "FALSIFIED" if med <= 0.3 or p_ind < 0.05 else "INCONCLUSIVE")
            res |= {"lopo_spearman": rhos, "median_lopo_spearman": med, "paradigm_indicator_F": Fv,
                    "paradigm_indicator_p": p_ind, "verdict": verdict}
        out[lab] = res
    # theory check: crit > 2/pi predicts gain(spectral) > gain(sign)
    tc = [(r["crit"] > 2 / math.pi, np.mean(list(r["y"]["vs_sign"].values())) > 0) for r in data
          if np.isfinite(r["crit"]) and r["y"]["vs_sign"]]
    if tc:
        pred, truth = np.array(tc).T
        out["theory_crit_vs_sign"] = {"n": len(tc), "accuracy": float((pred == truth).mean()),
                                      "majority_baseline": float(max(truth.mean(), 1 - truth.mean())),
                                      "frac_pred_spectral": float(pred.mean())}
    with open(run_dir / "h1_rows.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["paradigm", "source", "ckpt", "block"] + FEATURES + [f"y_{l}_{d}" for l in LABELS for d in (901, 902)])
        for r in data:
            w.writerow([r["paradigm"], r["source"], r["ckpt"], r["block"]] + [r["x"][k] for k in FEATURES]
                       + [r["y"][l].get(d, "") for l in LABELS for d in (901, 902)])
    return out
