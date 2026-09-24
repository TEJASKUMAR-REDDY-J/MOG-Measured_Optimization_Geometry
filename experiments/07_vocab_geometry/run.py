"""Exp 440 -- H5: per-token (row) geometry on vocabulary blocks vs Adam, with a real
rare-token regime (GPT-2 BPE remapped to the ~11.7k types in the corpus).

Protocol (pre-registered in EXPERIMENT_LOG): for each arm, tune LR on seed 0 over the
same 3-point grid, then evaluate the best LR on seeds 1-3. Report val loss and loss per
train-frequency decile of the target token (decile 0 = rarest).
"""

from __future__ import annotations

import copy
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from mog.training.trainer import get_data, train
from mog.utils.reproducibility import REPO_ROOT


@torch.no_grad()
def strata(model, data, batches, n_bins: int = 10):
    model.eval()
    ce, cnt = [], []
    for x, y in batches:
        ce.append(F.cross_entropy(model(x).flatten(0, 1), y.flatten(), reduction="none"))
        cnt.append(data.token_counts[y.flatten()])
    ce, cnt = torch.cat(ce), torch.cat(cnt)
    order = torch.argsort(cnt, stable=True)  # rarest first
    return [{"decile": i, "loss": float(ce[ix].mean()), "median_train_count": float(cnt[ix].median())}
            for i, ix in enumerate(torch.chunk(order, n_bins))]


def main(cfg: dict, run_dir: Path) -> dict:
    arms = yaml.safe_load(open(REPO_ROOT / cfg["arms_file"], encoding="utf-8"))["arms"]
    data = get_data("bpe")
    ev = data.eval_batches(cfg["train"]["eval_batches"], cfg["train"]["batch_size"], cfg["train"]["model"]["ctx"])
    hist_rows, strata_rows, summary = [], [], {"vocab": data.vocab}

    def run(arm, lr, seed, phase):
        tcfg = copy.deepcopy(cfg["train"]) | {"data": "bpe", "rules": arms[arm]["rules"], "lr": lr, "seed": seed}
        r = train(tcfg, log=print)
        hist_rows.extend({"phase": phase, "arm": arm, "lr": lr, "seed": seed, **h} for h in r["history"])
        s = [] if r["diverged"] else strata(r["model"], data, ev)
        strata_rows.extend({"phase": phase, "arm": arm, "lr": lr, "seed": seed, **d} for d in s)
        print(f"{phase:5s} {arm:8s} lr={lr:<9g} seed={seed} val={r['final_val']:.4f}", flush=True)
        return r["final_val"], s

    for arm in cfg["arms"]:
        tune = {lr: run(arm, lr, 0, "tune")[0] for lr in cfg["lr_grid"]}
        best = min(tune, key=tune.get)
        evals = {s: run(arm, best, s, "eval") for s in cfg["eval_seeds"]}
        summary[arm] = {"tune": tune, "lr": best, "val": {s: v for s, (v, _) in evals.items()},
                        "strata": {s: st for s, (_, st) in evals.items()}}
    for name, rows in (("metrics.csv", hist_rows), ("strata.csv", strata_rows)):
        with open(run_dir / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return summary
