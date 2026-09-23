"""Collect the latest run of each experiment into results/processed/report_data.json.

    python -m reports.build_report_data

Pure aggregation of results/raw/*: no numbers are produced here that are not in a
run directory, except summary statistics (means, std, paired t-intervals).
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict

import numpy as np
import yaml

from mog.models.tiny_transformer import GPT, GPTConfig, blocks
from mog.optim.blockwise import resolve_rules
from mog.utils.reproducibility import REPO_ROOT

RAW = REPO_ROOT / "results" / "raw"
T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def latest(exp_id):
    d = RAW / exp_id
    if not d.exists():
        return None
    runs = [r for r in sorted(d.iterdir()) if (r / "results.json").exists()
            and json.loads((r / "results.json").read_text())["status"] == "ok"]
    return runs[-1] if runs else None


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def sweep_summary(run):
    rows = read_csv(run / "metrics.csv")
    finals, curves = defaultdict(dict), defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["arm"], float(r["lr"]))
        finals[key][int(r["seed"])] = (float(r["val_loss"]), float(r["train_loss"]))
        curves[key][int(r["step"])].append((float(r["val_loss"]), float(r["train_loss"])))
    res = json.loads((run / "results.json").read_text())["summary"]
    for k, v in res.items():
        arm, lr, seed = k.split("|")
        if v["diverged"]:
            finals[(arm, float(lr))][int(seed)] = (math.inf, math.inf)
    return finals, curves


def paired(a: dict, b: dict):
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds])
    if len(d) < 2 or not np.all(np.isfinite(d)):
        return None
    half = T_CRIT[len(d) - 1] * d.std(ddof=1) / math.sqrt(len(d))
    return {"mean": float(d.mean()), "ci95": [float(d.mean() - half), float(d.mean() + half)], "n": len(d)}


def clean(o):
    """JSON has no Infinity/NaN: map non-finite floats (diverged runs) to None."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


def main():
    out = {"generated_from": {}}
    arms_spec = yaml.safe_load(open(REPO_ROOT / "configs" / "arms.yaml", encoding="utf-8"))["arms"]

    # network layout + per-arm rule map (for the diagram)
    model = GPT(GPTConfig(vocab=65, d=128, n_layer=4, n_head=4, ctx=128))
    bl = blocks(model)
    out["network"] = [{"name": b["name"], "kind": b["kind"], "layer": b["layer"], "shape": list(b["param"].shape)}
                      for b in bl]
    out["arm_rules"] = {a: resolve_rules(bl, s["rules"]) for a, s in arms_spec.items()}

    # Exp 0
    r0 = latest("exp000_synthetic_gradient_fit")
    if r0:
        out["generated_from"]["exp000"] = str(r0.relative_to(REPO_ROOT))
        s = json.loads((r0 / "results.json").read_text())["summary"]
        rows = read_csv(r0 / "metrics.csv")
        fig1b = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r["k"] == "":
                fig1b[f"{r['geometry']}|{r['method']}"][float(r["beta"])].append(float(r["theta_over_C"]))
        out["exp000"] = {"table": s["table_4_3"], "grouped": s["grouped_fig1c_mean"], "ns_fidelity": s["ns_fidelity"],
                         "fig1b": {k: {str(b): float(np.mean(v)) for b, v in sorted(d.items())} for k, d in fig1b.items()}}

    # Stage 2 tiny sweep
    r10 = latest("exp010_tiny_optimizer_sanity")
    if r10:
        out["generated_from"]["exp010"] = str(r10.relative_to(REPO_ROOT))
        finals, _ = sweep_summary(r10)
        out["exp010"] = [{"arm": a, "lr": lr, "val_mean": float(np.mean([v[0] for v in d.values()]))}
                         for (a, lr), d in sorted(finals.items())]

    # Stage 3 tuning + baselines + MOG arms
    r150, r160, r300 = latest("exp150_1m_lr_tuning"), latest("exp160_1m_baselines"), latest("exp300_mog_1m")
    if r150:
        out["generated_from"]["exp150"] = str(r150.relative_to(REPO_ROOT))
        finals, _ = sweep_summary(r150)
        out["exp150"] = [{"arm": a, "lr": lr, "val": d.get(0, (None,))[0]} for (a, lr), d in sorted(finals.items())]
    arms = {}
    for run in (r160, r300):
        if not run:
            continue
        out["generated_from"][run.parent.name] = str(run.relative_to(REPO_ROOT))
        finals, curves = sweep_summary(run)
        for (a, lr), d in finals.items():
            vals = [v[0] for v in d.values()]
            arms[a] = {"lr": lr, "val": {s: v[0] for s, v in d.items()}, "train": {s: v[1] for s, v in d.items()},
                       "val_mean": float(np.mean(vals)), "val_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                       "train_mean": float(np.mean([v[1] for v in d.values()])),
                       "curve": [{"step": st, "val": float(np.mean([x[0] for x in xs])), "train": float(np.mean([x[1] for x in xs])),
                                  "val_min": float(np.min([x[0] for x in xs])), "val_max": float(np.max([x[0] for x in xs]))}
                                 for st, xs in sorted(curves[(a, lr)].items())]}
            if run == r300 and (run / "mog_oracle_plan.json").exists():
                out["mog_oracle_plan"] = json.loads((run / "mog_oracle_plan.json").read_text())
    for a in arms:
        for ref in ("muon", "adamw"):
            if ref in arms and a != ref:
                arms[a][f"vs_{ref}"] = paired(arms[a]["val"], arms[ref]["val"])
    out["arms"] = arms

    # Atlas
    r100 = latest("exp100_atlas_1m")
    if r100:
        out["generated_from"]["exp100"] = str(r100.relative_to(REPO_ROOT))
        keep = ["arm", "step", "block", "kind", "layer", "rule", "r_eff_frac", "d_eff_frac", "m_eff_frac", "ns_dual_rel_err",
                "ns_cos", "argmax_free", "argmax_same", "snr_block", "snr_row_median", "act_pr"]
        rows = read_csv(r100 / "atlas.csv")
        out["atlas"] = []
        for r in rows:
            e = {k: (fnum(r[k]) if fnum(r[k]) is not None else r[k]) for k in keep}
            e["fit"] = {k[4:]: float(v) for k, v in r.items() if k.startswith("fit_") and v}
            e["Efree"] = {k[6:]: float(v) for k, v in r.items() if k.startswith("Efree_") and v}
            e["Esame"] = {k[6:]: float(v) for k, v in r.items() if k.startswith("Esame_") and v}
            e["Lratio"] = {k[6:]: float(v) / float(r["Lsame_" + k[6:]]) for k, v in r.items()
                           if k.startswith("Lfree_") and v and float(r["Lsame_" + k[6:]]) > 0}
            out["atlas"].append(e)
        out["atlas_training"] = json.loads((r100 / "results.json").read_text())["summary"]

    # Oracle
    r200 = latest("exp200_oracle_1m")
    if r200:
        out["generated_from"]["exp200"] = str(r200.relative_to(REPO_ROOT))
        rows = read_csv(r200 / "oracle.csv")
        cells = defaultdict(dict)
        for r in rows:
            c = cells[(int(r["ckpt"]), r["block"])]
            c.update(ckpt=int(r["ckpt"]), block=r["block"], kind=r["kind"], layer=int(r["layer"]), incumbent=r["incumbent"])
            c.setdefault("gains", {})[r["rule"]] = float(r["gain_vs_incumbent"])
            c.setdefault("proxy", {})[r["rule"]] = {k[6:]: float(v) for k, v in r.items() if k.startswith("proxy_")}
            if r["is_best"] == "True":
                c["best"], c["best_gain_alt"] = r["rule"], float(r["best_gain_alt"])
        out["oracle"] = list(cells.values())
        out["oracle_summary"] = json.loads((r200 / "results.json").read_text())["summary"]

    dst = REPO_ROOT / "reports" / "report_data.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(clean(out), allow_nan=False))
    print(f"wrote {dst} ({dst.stat().st_size // 1024} KB); sources: {out['generated_from']}")


if __name__ == "__main__":
    main()
