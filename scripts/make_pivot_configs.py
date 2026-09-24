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


if __name__ == "__main__" and not sys.argv[1].startswith("p2"):
    main(sys.argv[1])


# ---------------- Phase 2: four paradigms (pre-registered in docs/EXPERIMENT_LOG.md, "Pivot Phase 2")
CORE = ["muon", "adamw", "adam_rms", "randspec"]
P2 = {  # tag: (id base, arms, train cfg, lr grid, checkpoint steps, oracle horizons (class, block), groups)
    "sup": (600, CORE, {"task": "supervised", "steps": 1500, "warmup": 100, "min_lr_frac": 0.1, "eval_every": 250},
            [0.001, 0.003, 0.01], [375, 750, 1250], ([20, 100], [20]), {"hidden": "role:hidden", "boundary": "role:boundary"}),
    "ssl": (610, CORE + ["ssl_enc_muon", "ssl_proj_muon"],
            {"task": "ssl", "steps": 1000, "warmup": 100, "min_lr_frac": 0.1, "eval_every": 250},
            [0.001, 0.003, 0.01], [250, 500, 850], ([20, 100], [20]),
            {"hidden": "role:hidden", "boundary": "role:boundary", "encoder": "name:^trunk[.]", "projector": "name:^proj"}),
    "flow": (620, CORE, {"task": "flow", "steps": 3000, "warmup": 200, "min_lr_frac": 0.1, "eval_every": 500},
             [0.001, 0.003, 0.01], [750, 1500, 2500], ([20, 100], [20]), {"hidden": "role:hidden", "boundary": "role:boundary"}),
    "rl": (630, CORE, {"task": "rl", "env": "Breakout-MinAtar", "total_steps": 1048576, "eval_every": 32},
           [0.0003, 0.001, 0.003], [32, 64, 106], ([2, 8], [2]),
           {"hidden": "role:hidden", "boundary": "role:boundary", "actor": "name:^actor[.]", "critic": "name:^critic[.]"}),
}
CANDIDATES = ["spectral", "polar_svd", "randspec", "adam_rms", "adam", "sign", "rows"]
CONTRASTS = [["spectral", "adam_rms"], ["spectral", "randspec"], ["polar_svd", "randspec"], ["adam_rms", "adam"],
             ["spectral", "sign"], ["spectral", "rows"], ["spectral", "adam"]]
OUT2 = REPO_ROOT / "configs/pivot/p2"


def p2(mode):
    for tag, (base, arms, train, grid, ckpts, (hc, hb), groups) in P2.items():
        ids = {k: f"exp{base + i}_p2_{tag}_{k}" for i, k in enumerate(("tuning", "edge", "eval", "ckpts", "oracle", "stats"))}
        if train["task"] == "rl":
            train = train | {"checkpoints": []}
        if mode == "p2_tune":
            dump(OUT2 / f"{ids['tuning']}.yaml", sweep_cfg(ids["tuning"], f"Pivot Phase 2 ({tag}): 3-point LR grid, seed 0.",
                                                        arms, {a: list(grid) for a in arms}, [0], train) | {"entry": "experiments/09_paradigms/task_sweep.py"})
        elif mode == "p2_extend":
            best = best_lr(latest(ids["tuning"]), arms)
            ext = {a: [round(b / 3, 8)] if b == g[0] else [round(b * 3, 8)] for a, (b, g) in best.items() if b in (g[0], g[-1])}
            print(tag, "edge extensions:", ext)
            if ext:
                dump(OUT2 / f"{ids['edge']}.yaml", sweep_cfg(ids["edge"], f"Pivot Phase 2 ({tag}): edge rule.", sorted(ext), ext, [0], train)
                     | {"entry": "experiments/09_paradigms/task_sweep.py"})
        elif mode == "p2_eval":
            f = dict(sweep_summary(latest(ids["tuning"]))[0])
            if latest(ids["edge"]):
                f.update(sweep_summary(latest(ids["edge"]))[0])
            tuned = {}
            for a in arms:
                c = {lr: d[0][0] for (arm, lr), d in f.items() if arm == a and np.isfinite(d[0][0])}
                tuned[a] = [min(c, key=c.get)]
            print(tag, "tuned:", tuned)
            ent = {"entry": "experiments/09_paradigms/task_sweep.py"}
            dump(OUT2 / f"{ids['eval']}.yaml", sweep_cfg(ids["eval"], f"Pivot Phase 2 ({tag}): tuned LRs, seeds 1-5.",
                                                      arms, tuned, [1, 2, 3, 4, 5], train) | ent)
            dump(OUT2 / f"{ids['ckpts']}.yaml", sweep_cfg(ids["ckpts"], f"Pivot Phase 2 ({tag}): muon and adamw incumbents "
                                                       "at tuned LR, fresh seed 6, writing oracle checkpoints.",
                                                       ["muon", "adamw"], {a: tuned[a] for a in ("muon", "adamw")}, [6],
                                                       train | {"checkpoints": ckpts}) | ent | {"save_checkpoints": True})
            srcs = {a: f"latest:{ids['ckpts']}/{a}_s6" for a in ("muon", "adamw")}
            dump(OUT2 / f"{ids['oracle']}.yaml", {
                "experiment_id": ids["oracle"], "entry": "experiments/09_paradigms/oracle_tasks.py", "evidence_class": "measured",
                "description": f"Pivot Phase 2 ({tag}): paired unit oracle (classes and single blocks) from muon and adamw incumbents.",
                "seed": 6, "sources": srcs, "checkpoints": ckpts, "candidates": CANDIDATES, "groups": groups,
                "class_units": {"horizons": hc, "data_seeds": [901, 902]},
                "block_units": {"sources": ["muon", "adamw"], "horizons": hb, "data_seeds": [901, 902]},
                "contrasts": CONTRASTS, "workers": 4})
            dump(OUT2 / f"{ids['stats']}.yaml", {
                "experiment_id": ids["stats"], "entry": "experiments/09_paradigms/block_stats.py", "evidence_class": "measured",
                "description": f"Pivot Phase 2 ({tag}): block statistics (H1 predictors) at the oracle checkpoints.",
                "seed": 6, "sources": srcs, "checkpoints": ckpts, "K": 16, "probe_seed": 5151, "workers": 4})


if __name__ == "__main__" and sys.argv[1].startswith("p2"):
    p2(sys.argv[1])
