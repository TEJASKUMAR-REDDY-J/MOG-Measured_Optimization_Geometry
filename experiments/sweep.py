"""Generic arm x LR x seed sweep (used by exp010, exp150, exp160, exp300).

Config:
  arms_file: configs/arms.yaml
  arms: [names]                      # subset of arms_file (or MOG arms, see below)
  lr: {arm: [lrs]} | "grid"          # "grid" -> lr_grids[family] from arms_file
  seeds: [..]
  train: {...trainer cfg without rules/lr/seed...}
  mog_oracle: {oracle_run: results/raw/..., incumbent_arm: muon}   # optional
Writes metrics.csv (appended after every run) and returns per-(arm, lr, seed) finals.
"""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import yaml

from mog.training.trainer import train
from mog.utils.reproducibility import REPO_ROOT

FIELDS = ["arm", "lr", "seed", "step", "train_loss", "val_loss", "lr_now", "wall_s"]


def oracle_schedule(oracle_run: str):
    """Piecewise rule schedule from Exp 2: at step s use the assignment of the latest
    oracle checkpoint <= s (the first checkpoint's assignment before it). Only switches
    that were stable on the alternate data stream are applied."""
    rows = list(csv.DictReader(open(REPO_ROOT / oracle_run / "oracle.csv", encoding="utf-8")))
    plan = {}
    for r in rows:
        if r["is_best"] == "True" and r["rule"] != r["incumbent"] and r["best_gain_alt"] and float(r["best_gain_alt"]) > 0:
            plan.setdefault(int(r["ckpt"]), {})[r["block"]] = r["rule"]
    ckpts = sorted({int(r["ckpt"]) for r in rows})
    plan = {c: plan.get(c, {}) for c in ckpts}

    def schedule(step, opt, _base={}):
        if not _base:
            _base.update(opt.rules)
        active = max([c for c in ckpts if c <= step], default=ckpts[0])
        for name, rule in _base.items():
            opt.set_rule(name, plan[active].get(name, rule))
    return schedule, plan


def main(cfg: dict, run_dir: Path) -> dict:
    spec = yaml.safe_load(open(REPO_ROOT / cfg["arms_file"], encoding="utf-8"))
    arms, grids = spec["arms"], spec["lr_grids"]
    out = {}
    with open(run_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    for arm in cfg["arms"]:
        sched, base_arm = None, arm
        if arm == "mog_oracle":
            base_arm = cfg["mog_oracle"]["incumbent_arm"]
            sched, plan = oracle_schedule(cfg["mog_oracle"]["oracle_run"])
            (run_dir / "mog_oracle_plan.json").write_text(json.dumps(plan, indent=2))
        lrs = grids[arms[base_arm]["family"]] if cfg["lr"] == "grid" else cfg["lr"][arm]
        for lr in lrs:
            for seed in cfg["seeds"]:
                tcfg = copy.deepcopy(cfg["train"]) | {"rules": arms[base_arm]["rules"], "lr": lr, "seed": seed}
                r = train(tcfg, rule_schedule=sched, log=print)
                with open(run_dir / "metrics.csv", "a", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=FIELDS)
                    for h in r["history"]:
                        w.writerow({"arm": arm, "lr": lr, "seed": seed, "step": h["step"], "train_loss": h["train_loss"],
                                    "val_loss": h["val_loss"], "lr_now": h["lr"], "wall_s": h["wall_s"]})
                out[f"{arm}|{lr}|{seed}"] = {"final_val": r["final_val"], "final_train": r["final_train"],
                                             "diverged": r["diverged"], "wall_s": r["wall_s"]}
                print(f"{arm:10s} lr={lr:<8g} seed={seed} val={r['final_val']:.4f} train={r['final_train']:.4f} "
                      f"{'DIVERGED ' if r['diverged'] else ''}({r['wall_s']} s)", flush=True)
    return out
