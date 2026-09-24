"""End-to-end arm comparisons with the pre-registered readings (docs/EXPERIMENT_LOG.md, pivot
Phase 0 / Phase 2). Analysis only.

Config: studies {name: {eval: <run spec>, margin: <float> | "rel10"}}, where "rel10" means
10% of |mean(adamw) - mean(muon)| in that study. Contrasts (paired over evaluation seeds):
    C1 muon - adam_rms   C2 muon - randspec   C3 polar - randspec   C4 adamw - adam_rms
    C5 muon - spec_adamscale
(values are loss differences: negative = first arm better). Contrasts whose arms are absent are skipped.
Readings: (a) C1 and (C2 or C3) favour spectral with CI excluding 0; (b) C1 favours spectral and
C2/C3 TOST-equivalent; (c) C1 and C2 TOST-equivalent; (d) otherwise.
Also reports final_metrics per arm (probe accuracy / RankMe for SSL, per-t-bin loss for flow).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from mog.selection.oracle import paired_ci, spearman, tost
from mog.utils.reproducibility import REPO_ROOT

CONTRASTS = {"C1": ("muon", "adam_rms"), "C2": ("muon", "randspec"), "C3": ("polar", "randspec"),
             "C4": ("adamw", "adam_rms"), "C5": ("muon", "spec_adamscale")}


def _run(spec):
    exp = spec.split(":", 1)[1] if spec.startswith("latest:") else spec
    return sorted((REPO_ROOT / "results" / "raw" / exp).iterdir())[-1]


def finals(run):
    res = json.loads((run / "results.json").read_text())["summary"]
    out = {}
    for k, v in res.items():
        arm, _, seed = k.split("|")
        out.setdefault(arm, {})[int(seed)] = v
    return out


def study(spec) -> dict:
    run = _run(spec["eval"])
    F = finals(run)
    val = {a: {s: v["final_val"] for s, v in d.items()} for a, d in F.items()}
    out = {"run": str(run.relative_to(REPO_ROOT)),
           "arms": {a: {"mean": float(np.mean(list(d.values()))), "sd": float(np.std(list(d.values()), ddof=1)),
                        "n": len(d), "per_seed": d} for a, d in val.items()}}
    margin = spec["margin"]
    if margin == "rel10":
        margin = 0.1 * abs(out["arms"]["adamw"]["mean"] - out["arms"]["muon"]["mean"])
    out["margin"] = margin
    con = {}
    for c, (a, b) in CONTRASTS.items():
        if a in val and b in val:
            seeds = sorted(set(val[a]) & set(val[b]))
            d = np.array([val[a][s] - val[b][s] for s in seeds])
            if np.all(np.isfinite(d)):
                con[c] = paired_ci(d) | {"tost": tost(d, margin), "arms": [a, b]}
    out["contrasts"] = con

    def favours(c):
        return c in con and "ci95" in con[c] and con[c]["ci95"][1] < 0

    def equiv(c):
        return c in con and con[c]["tost"]["equivalent"]

    if favours("C1") and (favours("C2") or favours("C3")):
        out["reading"] = "(a) shape matters"
    elif favours("C1") and (equiv("C2") or equiv("C3")):
        out["reading"] = "(b) singular vectors matter, singular values do not"
    elif equiv("C1") and equiv("C2"):
        out["reading"] = "(c) scale only"
    else:
        out["reading"] = "(d) inconclusive"
    fm = {a: [v.get("final_metrics") for v in d.values() if v.get("final_metrics")] for a, d in F.items()}
    out["final_metrics"] = {a: {k: float(np.mean([m[k] for m in ms])) for k in ms[0]} for a, ms in fm.items() if ms}
    return out


def h6(s) -> dict:
    """SSL 2x2: projector effect vs encoder effect on probe accuracy (higher = better)."""
    F = finals(_run(s["eval"]))
    acc = {a: {sd: v["final_metrics"]["probe_acc"] for sd, v in d.items()} for a, d in F.items()}
    enc = [acc["ssl_enc_muon"][k] - acc["adamw"][k] for k in acc["adamw"]]
    proj = [acc["ssl_proj_muon"][k] - acc["adamw"][k] for k in acc["adamw"]]
    diff = np.array(proj) - np.array(enc)
    return {"encoder_effect": paired_ci(enc), "projector_effect": paired_ci(proj), "projector_minus_encoder": paired_ci(diff),
            "verdict": "FALSIFIED (encoder effect >= projector effect)" if np.mean(diff) <= 0 else
                       ("SUPPORTED" if paired_ci(diff)["ci95"][0] > 0 else "direction as predicted, CI includes 0")}


def h7(s) -> dict:
    """Flow: muon advantage over adamw per t-bin (loss difference adamw - muon, >0 = muon better)."""
    F = finals(_run(s["eval"]))
    bins = sorted(k for k in next(iter(F["muon"].values()))["final_metrics"] if k.startswith("t"))
    adv = [float(np.mean([F["adamw"][sd]["final_metrics"][b] - F["muon"][sd]["final_metrics"][b] for sd in F["muon"]])) for b in bins]
    rho = spearman(np.arange(len(bins)), np.array(adv))
    return {"advantage_by_t_bin": dict(zip(bins, adv)), "spearman_vs_t": rho,
            "verdict": "direction as predicted (advantage grows with t)" if rho > 0 else "FALSIFIED (advantage does not grow with t)"}


def h5(spec) -> dict:
    """RL actor-critic: critic.h gradient nuclear rank < actor.h at every (source, checkpoint); and the
    single-block spectral - adam gain is no larger on critic blocks than on actor blocks."""
    st = list(csv.DictReader(open(_run(spec["stats"]) / "block_stats.csv", encoding="utf-8")))
    cells = {}
    for r in st:
        cells.setdefault((r["source"], r["ckpt"]), {})[r["block"]] = float(r["nuc_rank_g"])
    rank_ok = [c["critic.h"] < c["actor.h"] for c in cells.values()]
    orc = [r for r in csv.DictReader(open(_run(spec["oracle"]) / "oracle_units.csv", encoding="utf-8")) if r["unit_type"] == "block"]
    g = {}
    for r in orc:
        g.setdefault((r["source"], r["ckpt"], r["unit"], r["data_seed"]), {})[r["rule"]] = float(r["gain"])
    d = {"actor": [], "critic": []}
    for (_, _, unit, _), v in g.items():
        d[unit.split(".")[0]].append(v.get("spectral", 0.0) - v.get("adam", 0.0))
    return {"critic_rank_below_actor": f"{sum(rank_ok)}/{len(rank_ok)}",
            "spectral_minus_adam_gain": {k: paired_ci(v) for k, v in d.items()},
            "verdict": "SUPPORTED" if all(rank_ok) and np.mean(d["critic"]) <= np.mean(d["actor"]) else "FALSIFIED"}


def main(cfg: dict, run_dir: Path) -> dict:
    out = {name: study(s) for name, s in cfg["studies"].items()}
    if "h5" in cfg:
        out["H5"] = h5(cfg["h5"])
    for name, s in cfg["studies"].items():
        if s.get("h6"):
            out[name]["H6"] = h6(s)
        if s.get("h7"):
            out[name]["H7"] = h7(s)
    for name, r in out.items():
        if name == "H5":
            continue
        print(name, r["reading"], {c: round(v["mean"], 4) for c, v in r["contrasts"].items()})
    return out
