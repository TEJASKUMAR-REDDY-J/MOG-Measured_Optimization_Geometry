"""Write pivot (cross-paradigm) configs from earlier results, by the pre-registered rules in
docs/EXPERIMENT_LOG.md ("Pivot Phase 0").

    python -m scripts.make_pivot_configs p0_tune      # exp500 (char) + exp510 (BPE) tuning grids
    python -m scripts.make_pivot_configs p0_extend    # exp501 / exp511 edge extensions
    python -m scripts.make_pivot_configs p0_eval      # exp502 / exp512 evaluation at tuned LRs, seeds 1-5
"""

from __future__ import annotations

import sys

import numpy as np

from reports.build_report_data import latest, sweep_summary
from scripts.make_stage3_configs import TRAIN_1M, best_lr, dump
from mog.utils.reproducibility import REPO_ROOT

ARMS = ["muon", "adamw", "adam_rms", "spec_adamscale", "polar", "randspec", "kaon", "freon23", "freon34",
        "sign_h", "rows_h"]
ARMS_BPE = ["muon", "adamw", "adam_rms", "polar", "randspec", "kaon"]
TRAIN = TRAIN_1M | {"threads": 1}
TRAIN_BPE = TRAIN | {"data": "bpe", "steps": 300, "warmup": 30, "eval_every": 50, "eval_batches": 16}
GRID = [0.00333333, 0.01, 0.03]
SETS = {"char": ("exp500_p0_tuning", "exp501_p0_edge", "exp502_p0_eval", ARMS, TRAIN),
        "bpe": ("exp510_p0_bpe_tuning", "exp511_p0_bpe_edge", "exp512_p0_bpe_eval", ARMS_BPE, TRAIN_BPE)}
OUT = REPO_ROOT / "configs/pivot"


def sweep_cfg(eid, desc, arms, lr, seeds, train):
    return {"experiment_id": eid, "entry": "experiments/sweep.py", "evidence_class": "measured", "description": desc,
            "seed": seeds[0], "arms_file": "configs/arms.yaml", "arms": arms, "lr": lr, "seeds": seeds,
            "workers": 4, "train": train}


def main(mode):
    for tag, (tune, edge, ev, arms, train) in SETS.items():
        if mode == "p0_tune":
            dump(OUT / f"{tune}.yaml", sweep_cfg(tune, f"Pivot Phase 0 ({tag}): 3-point LR grid, seed 0 only.",
                                                 arms, {a: GRID for a in arms}, [0], train))
        elif mode == "p0_extend":
            best = best_lr(latest(tune), arms)
            ext = {a: [round(b / 3, 8)] if b == g[0] else [round(b * 3, 8)] for a, (b, g) in best.items()
                   if b in (g[0], g[-1])}
            print(tag, "edge extensions:", ext)
            if ext:
                dump(OUT / f"{edge}.yaml", sweep_cfg(edge, f"Pivot Phase 0 ({tag}): pre-registered edge rule.",
                                                     sorted(ext), ext, [0], train))
        elif mode == "p0_eval":
            f = dict(sweep_summary(latest(tune))[0])
            if latest(edge):
                f.update(sweep_summary(latest(edge))[0])
            tuned = {}
            for a in arms:
                c = {lr: d[0][0] for (arm, lr), d in f.items() if arm == a and np.isfinite(d[0][0])}
                tuned[a] = [min(c, key=c.get)]
            print(tag, "tuned:", tuned)
            dump(OUT / f"{ev}.yaml", sweep_cfg(ev, f"Pivot Phase 0 ({tag}): tuned LRs, evaluation seeds 1-5.",
                                               arms, tuned, [1, 2, 3, 4, 5], train))


if __name__ == "__main__":
    main(sys.argv[1])
