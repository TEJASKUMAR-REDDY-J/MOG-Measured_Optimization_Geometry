"""Oracle geometry assignment (Exp 2).

The runnable implementation is experiments/03_oracle/oracle.py: single-block switch,
h-step rollouts from a frozen checkpoint on a shared data stream, and an alternate-stream
stability check. It moves here behind a model-agnostic
`oracle(checkpoint, loss_fn, data_stream, candidates, horizon)` API after Stage 3
(docs/FRAMEWORK.md, step 2). It is not moved now because Stage 3 runs are executing it.
"""

import numpy as np


def spearman(a, b) -> float:
    """Rank correlation used to score proxies against the oracle (ties broken by order)."""
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def paired_ci(d) -> dict:
    """Mean, 95% t-interval and count of positive paired differences."""
    from scipy import stats

    d = np.asarray(d, float)
    n, m = len(d), float(np.mean(d)) if len(d) else float("nan")
    if n < 2:
        return {"n": n, "mean": m}
    se = float(np.std(d, ddof=1) / np.sqrt(n))
    h = float(stats.t.ppf(0.975, n - 1) * se) if se > 0 else 0.0
    return {"n": n, "mean": m, "ci95": [m - h, m + h], "n_pos": int((d > 0).sum())}


def tost(d, margin: float, alpha: float = 0.05) -> dict:
    """Two one-sided t-tests for |mean difference| < margin. Equivalent iff p < alpha."""
    from scipy import stats

    d = np.asarray(d, float)
    n = len(d)
    se = float(np.std(d, ddof=1) / np.sqrt(n))
    if se == 0:
        return {"p": 0.0 if abs(d.mean()) < margin else 1.0, "equivalent": abs(d.mean()) < margin}
    p = max(stats.t.sf((d.mean() + margin) / se, n - 1), stats.t.sf((margin - d.mean()) / se, n - 1))
    return {"p": float(p), "equivalent": bool(p < alpha), "margin": margin}


def unit_contrasts(rows, candidates, contrasts, group_of) -> dict:
    """rows: dicts with source, unit, unit_type, horizon, ckpt, data_seed, rule, gain.
    Per (source, group, horizon): mean gain per rule (incumbent = 0) and paired contrasts."""
    cells = {}
    for r in rows:
        key = (r["source"], group_of(r), int(r["horizon"]))
        cells.setdefault(key, {}).setdefault((r["ckpt"], r["unit"], r["data_seed"]), {})[r["rule"]] = float(r["gain"])
    out = {}
    for (src, grp, h), pairs in sorted(cells.items()):
        for p in pairs.values():
            for c in candidates:
                p.setdefault(c, 0.0)  # missing = the incumbent itself
        out[f"{src}|{grp}|h{h}"] = {
            "mean_gain": {c: float(np.mean([p[c] for p in pairs.values()])) for c in candidates},
            "contrasts": {f"{a}-{b}": paired_ci([p[a] - p[b] for p in pairs.values() if np.isfinite(p[a]) and np.isfinite(p[b])])
                          for a, b in contrasts},
            "n_pairs": len(pairs)}
    return out
