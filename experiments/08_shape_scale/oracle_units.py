"""Pivot Phase 0 -- shape vs scale, paired oracle on block classes and single blocks.

From each frozen checkpoint, switch a UNIT (a block class -- attn / mlp / hidden -- or one
hidden block) to candidate rule c, keep every other block on its incumbent rule, train h
steps on a fixed stream at the checkpoint's LR, and measure val loss. Same data for every
candidate -> paired. gain(c) = L(incumbent) - L(c).

Config:
  sources: {name: latest:<experiment_id>}     # checkpoint runs (incumbent = the run's rules)
  checkpoints: [100, 300, 500]
  candidates: [rules]
  class_units: {horizons: [20, 100], data_seeds: [901, 902]}
  block_units: {sources: [muon], horizons: [20], data_seeds: [901]}   # single hidden blocks
  contrasts: [[a, b], ...]                     # paired differences gain(a) - gain(b)
  workers: 4
"""

from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from mog.optim.blockwise import HIDDEN
from mog.training.trainer import rollout_job
from mog.utils.reproducibility import REPO_ROOT

CLASSES = {"attn": ("q", "k", "v", "o"), "mlp": ("up", "down"), "hidden": HIDDEN}


def _src(spec: str) -> Path:
    if spec.startswith("latest:"):
        return sorted((REPO_ROOT / "results" / "raw" / spec.split(":", 1)[1]).iterdir())[-1]
    return REPO_ROOT / spec


def paired_ci(d) -> dict:
    d = np.asarray(d, float)
    n, m = len(d), float(np.mean(d))
    if n < 2:
        return {"n": n, "mean": m}
    se = float(np.std(d, ddof=1) / np.sqrt(n))
    h = float(stats.t.ppf(0.975, n - 1) * se) if se > 0 else 0.0
    return {"n": n, "mean": m, "ci95": [m - h, m + h], "n_pos": int((d > 0).sum())}


def main(cfg: dict, run_dir: Path) -> dict:
    jobs = []  # (meta, ckpt_path, switches, horizon, data_seed)
    for sname, spec in cfg["sources"].items():
        src = _src(spec)
        print(f"source {sname}: {src.relative_to(REPO_ROOT)}")
        for s in cfg["checkpoints"]:
            path = str(src / "checkpoints" / f"step_{s}.pt")
            ck = torch.load(path, weights_only=False)
            rules = ck["opt"]["rules"]
            kinds = {n: n.split(".")[-1] for n in rules}
            units = [("class", c, [n for n in rules if kinds[n] in ks]) for c, ks in CLASSES.items()]
            if sname in cfg["block_units"]["sources"]:
                units += [("block", n, [n]) for n in rules if kinds[n] in HIDDEN]
            for utype, uname, members in units:
                u = cfg[f"{utype}_units"]
                inc = {rules[n] for n in members}
                for h in u["horizons"]:
                    for ds in u["data_seeds"]:
                        for rule in cfg["candidates"]:
                            if inc == {rule}:
                                continue  # candidate == incumbent -> gain 0 by definition
                            meta = {"source": sname, "ckpt": s, "unit_type": utype, "unit": uname,
                                    "horizon": h, "data_seed": ds, "rule": rule}
                            jobs.append((meta, path, {n: rule for n in members}, h, ds))
                        jobs.append(({"source": sname, "ckpt": s, "unit_type": "base", "unit": "-", "horizon": h,
                                      "data_seed": ds, "rule": "incumbent"}, path, {}, h, ds))
    # dedupe incumbent rollouts
    seen, uniq = set(), []
    for j in jobs:
        key = (j[1], tuple(sorted(j[2].items())), j[3], j[4])
        if key not in seen:
            seen.add(key)
            uniq.append(j)
    print(f"{len(uniq)} rollouts", flush=True)
    with ProcessPoolExecutor(cfg.get("workers", 4)) as pool:
        futs = [pool.submit(rollout_job, p, sw, h, ds) for _, p, sw, h, ds in uniq]
        losses = []
        for i, f in enumerate(futs):
            losses.append(f.result())
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(uniq)}", flush=True)
    base = {(m["source"], m["ckpt"], m["horizon"], m["data_seed"]): L
            for (m, *_), L in zip(uniq, losses) if m["unit_type"] == "base"}
    rows = []
    for (m, *_), L in zip(uniq, losses):
        if m["unit_type"] == "base":
            continue
        b = base[(m["source"], m["ckpt"], m["horizon"], m["data_seed"])]
        rows.append(m | {"loss": L, "base_loss": b, "gain": b - L})
    with open(run_dir / "oracle_units.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return analyze(rows, cfg)


def analyze(rows, cfg) -> dict:
    """Per (source, unit group, horizon): mean gain per rule and paired contrasts.
    Unit group = the class unit itself, or 'blocks:<class>' pooling single blocks of that class.
    Incumbent rule has gain 0 by definition (added explicitly)."""
    def group(r):
        if r["unit_type"] == "class":
            return r["unit"]
        kind = r["unit"].split(".")[-1]
        return "blocks:" + ("attn" if kind in CLASSES["attn"] else "mlp")

    cells = {}
    for r in rows:
        key = (r["source"], group(r), int(r["horizon"]))
        pair = (r["ckpt"], r["unit"], r["data_seed"])
        cells.setdefault(key, {}).setdefault(pair, {})[r["rule"]] = float(r["gain"])
    out = {}
    for (src, grp, h), pairs in sorted(cells.items()):
        # incumbent: every rule missing from a pair is the incumbent (gain 0)
        for p in pairs.values():
            for c in cfg["candidates"]:
                p.setdefault(c, 0.0)
        tag = f"{src}|{grp}|h{h}"
        out[tag] = {"mean_gain": {c: float(np.mean([p[c] for p in pairs.values()])) for c in cfg["candidates"]},
                    "contrasts": {f"{a}-{b}": paired_ci([p[a] - p[b] for p in pairs.values()
                                                        if np.isfinite(p[a]) and np.isfinite(p[b])])
                                  for a, b in cfg["contrasts"]},
                    "n_pairs": len(pairs)}
    return out
