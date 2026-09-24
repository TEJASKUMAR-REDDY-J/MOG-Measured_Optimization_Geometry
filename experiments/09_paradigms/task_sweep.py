"""Arm x LR x seed sweep for any task (GPT, torch paradigm tasks, JAX RL), in worker processes.

Config: arms_file, arms, lr {arm: [lrs]}, seeds, train {... incl. task}, workers,
optional save_checkpoints (each run writes checkpoints under run_dir/<arm>_s<seed>/).
metrics.csv has the same columns as experiments/sweep.py; results.json adds final_metrics.
"""

from __future__ import annotations

import copy
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

from mog.training.jobs import run_train
from mog.utils.reproducibility import REPO_ROOT

FIELDS = ["arm", "lr", "seed", "step", "train_loss", "val_loss", "lr_now", "wall_s"]
# one XLA thread per worker process (JAX otherwise spreads every process over all cores)
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")


def main(cfg: dict, run_dir: Path) -> dict:
    arms = yaml.safe_load(open(REPO_ROOT / cfg["arms_file"], encoding="utf-8"))["arms"]
    jobs = []
    for arm in cfg["arms"]:
        for lr in cfg["lr"][arm]:
            for seed in cfg["seeds"]:
                tcfg = copy.deepcopy(cfg["train"]) | {"rules": arms[arm]["rules"], "lr": lr, "seed": seed, "threads": 1}
                rd = None
                if cfg.get("save_checkpoints"):
                    rd = run_dir / f"{arm}_s{seed}"
                    rd.mkdir(exist_ok=True)
                jobs.append((arm, lr, seed, tcfg, rd))
    with open(run_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    out = {}
    with ProcessPoolExecutor(cfg.get("workers", 4)) as pool:
        futs = [pool.submit(run_train, j[3], j[4]) for j in jobs]
        for (arm, lr, seed, _, _), fut in zip(jobs, futs):
            r = fut.result()
            with open(run_dir / "metrics.csv", "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                for h in r["history"]:
                    w.writerow({"arm": arm, "lr": lr, "seed": seed, "step": h["step"], "train_loss": h["train_loss"],
                                "val_loss": h["val_loss"], "lr_now": h["lr"], "wall_s": h["wall_s"]})
            out[f"{arm}|{lr}|{seed}"] = {k: r[k] for k in ("final_val", "final_train", "diverged", "wall_s")} | (
                {"final_metrics": r["final_metrics"]} if "final_metrics" in r else {})
            print(f"{arm:14s} lr={lr:<9g} seed={seed} val={r['final_val']:.4f} {'DIVERGED ' if r['diverged'] else ''}"
                  f"({r['wall_s']} s)", flush=True)
    return out
