"""Exp 420 -- do single-block oracle switches add up, and do they survive a longer horizon?

Reads (never writes) the exp200 oracle plan and the exp100 checkpoints it came from.
At each checkpoint, on a fresh data stream, for horizons h in cfg["horizons"]:
    base   all blocks incumbent
    indiv  each planned (stable) switch alone
    joint  all planned switches of that checkpoint together
gain = L(base) - L(variant) on the fixed validation batches.
"""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import torch

from experiments.sweep import oracle_schedule
from mog.training.trainer import evaluate, get_data, lr_at, train
from mog.utils.reproducibility import REPO_ROOT


def rollout(ck, switches: dict, h: int, data_seed: int, ev) -> float:
    rcfg = copy.deepcopy(ck["cfg"])
    rcfg.update(steps=ck["step"] + h, data_seed=data_seed, eval_every=10**9,
                const_lr=lr_at(ck["step"] - 1, ck["cfg"]), checkpoints=[])

    def sched(step, opt):
        for block, rule in switches.items():
            opt.set_rule(block, rule)

    r = train(rcfg, start=ck, rule_schedule=sched if switches else None, log=lambda *a: None)
    return float("inf") if r["diverged"] else evaluate(r["model"], ev)


def main(cfg: dict, run_dir: Path) -> dict:
    oracle_dir = sorted((REPO_ROOT / "results" / "raw" / cfg["oracle_experiment"]).iterdir())[-1]
    src = REPO_ROOT / json.loads((oracle_dir / "results.json").read_text())["summary"]["source_run"].replace("\\", "/")
    _, plan = oracle_schedule(str(oracle_dir.relative_to(REPO_ROOT)))
    print(f"oracle: {oracle_dir}\ncheckpoints: {src}")
    data = get_data()
    rows, summary = [], {"oracle_run": str(oracle_dir.relative_to(REPO_ROOT)), "source_run": str(src.relative_to(REPO_ROOT))}
    for s, switches in plan.items():
        ck = torch.load(src / "checkpoints" / f"step_{s}.pt", weights_only=False)
        ev = data.eval_batches(ck["cfg"]["eval_batches"], ck["cfg"]["batch_size"], ck["cfg"]["model"]["ctx"])
        for h in cfg["horizons"]:
            base = rollout(ck, {}, h, cfg["data_seed"], ev)
            joint = base - rollout(ck, switches, h, cfg["data_seed"], ev)
            indiv = {b: base - rollout(ck, {b: r}, h, cfg["data_seed"], ev) for b, r in switches.items()}
            for b, g in indiv.items():
                rows.append({"ckpt": s, "horizon": h, "block": b, "rule": switches[b], "gain": g})
            rows.append({"ckpt": s, "horizon": h, "block": "JOINT", "rule": "", "gain": joint})
            tot = sum(indiv.values())
            summary[f"{s}|h{h}"] = {"n_switches": len(switches), "sum_indiv_gain": tot, "joint_gain": joint,
                                    "ratio_joint_to_sum": joint / tot if tot else None,
                                    "frac_indiv_positive": sum(g > 0 for g in indiv.values()) / max(1, len(indiv))}
            print(f"step {s} h={h}: {len(switches)} switches, sum indiv {tot:+.5f}, joint {joint:+.5f}", flush=True)
    with open(run_dir / "additivity.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return summary
