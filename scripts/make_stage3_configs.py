"""Write explicit Stage-3 configs from earlier results (the pre-registered rules in EXPERIMENT_LOG).

    python -m scripts.make_stage3_configs tuning     # exp150 grid from exp010 tiny-best LRs
    python -m scripts.make_stage3_configs eval       # exp160 / exp100 LRs from exp150 best (+ edge extensions)

The generated YAML files are committed, so every run is reproducible from its config alone.
"""

from __future__ import annotations

import sys

import numpy as np
import yaml

from mog.utils.reproducibility import REPO_ROOT
from scripts.build_report_data import latest, sweep_summary

ARMS = ["sgd", "adamw", "lion", "adam_mini", "signum", "normsgd", "rownorm", "muon_all",
        "muon", "normuon", "headmuon", "pw05", "duality", "scion"]
TRAIN_1M = {"model": {"d": 128, "n_layer": 4, "n_head": 4, "ctx": 128}, "batch_size": 16, "steps": 600,
            "warmup": 60, "min_lr_frac": 0.1, "eval_every": 100, "eval_batches": 8, "threads": 4}


def best_lr(run, arms):
    finals, _ = sweep_summary(run)
    best = {}
    for a in arms:
        cands = {lr: np.mean([v[0] for v in d.values()]) for (arm, lr), d in finals.items() if arm == a}
        cands = {lr: v for lr, v in cands.items() if np.isfinite(v)}
        best[a] = (min(cands, key=cands.get), sorted(cands)) if cands else (None, [])
    return best


def dump(path, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print("wrote", path.relative_to(REPO_ROOT))


def main(mode):
    if mode == "tuning":
        best = best_lr(latest("exp010_tiny_optimizer_sanity"), ARMS)
        lr = {a: [round(b * f, 8) for f in (1 / 3, 1, 3)] for a, (b, _) in best.items()}
        dump(REPO_ROOT / "configs/1m/exp150_lr_tuning.yaml", {
            "experiment_id": "exp150_1m_lr_tuning", "entry": "experiments/sweep.py", "evidence_class": "measured",
            "description": "Stage-3 LR tuning: 3-point grid {1/3,1,3} x exp010 tiny-best LR per arm, seed 0 only.",
            "seed": 0, "arms_file": "configs/arms.yaml", "arms": ARMS, "lr": lr, "seeds": [0], "train": TRAIN_1M})
    elif mode == "extend":
        best = best_lr(latest("exp150_1m_lr_tuning"), ARMS)
        ext = {}
        for a, (b, grid) in best.items():
            if b == grid[0]:
                ext[a] = [round(b / 3, 8)]
            elif b == grid[-1]:
                ext[a] = [round(b * 3, 8)]
        print("edge extensions:", ext)
        dump(REPO_ROOT / "configs/1m/exp151_lr_edge_extension.yaml", {
            "experiment_id": "exp151_1m_lr_edge_extension", "entry": "experiments/sweep.py", "evidence_class": "measured",
            "description": "Pre-registered edge rule: one extra LR beyond the edge for arms whose exp150 best sat at a grid edge.",
            "seed": 0, "arms_file": "configs/arms.yaml", "arms": sorted(ext), "lr": ext, "seeds": [0], "train": TRAIN_1M})
    elif mode == "eval":
        f150, _ = sweep_summary(latest("exp150_1m_lr_tuning"))
        r151 = latest("exp151_1m_lr_edge_extension")
        allf = dict(f150)
        if r151:
            allf.update(sweep_summary(r151)[0])
        tuned = {}
        for a in ARMS:
            c = {lr: d[0][0] for (arm, lr), d in allf.items() if arm == a and np.isfinite(d[0][0])}
            tuned[a] = min(c, key=c.get)
        print("tuned:", tuned)
        dump(REPO_ROOT / "configs/1m/exp160_baselines.yaml", {
            "experiment_id": "exp160_1m_baselines", "entry": "experiments/sweep.py", "evidence_class": "measured",
            "description": "All arms at their exp150/151-tuned LR, evaluation seeds 1-3 (seed 0 reserved for tuning/oracle).",
            "seed": 1, "arms_file": "configs/arms.yaml", "arms": ARMS, "lr": {a: [v] for a, v in tuned.items()},
            "seeds": [1, 2, 3], "train": TRAIN_1M})
        arms_yaml = yaml.safe_load(open(REPO_ROOT / "configs/arms.yaml"))["arms"]
        dump(REPO_ROOT / "configs/1m/exp100_atlas.yaml", {
            "experiment_id": "exp100_atlas_1m", "entry": "experiments/01_geometry_validation/atlas.py",
            "evidence_class": "measured",
            "description": "Geometry atlas on the 0.82M model: muon (incumbent, writes oracle checkpoints) and adamw, seed 0, tuned LRs.",
            "seed": 0, "arms": {"muon": arms_yaml["muon"]["rules"], "adamw": arms_yaml["adamw"]["rules"]},
            "lr": {"muon": tuned["muon"], "adamw": tuned["adamw"]}, "atlas_every": 50,
            "checkpoints": [100, 300, 500], "train": TRAIN_1M | {"seed": 0}})


    elif mode in ("mog_tuning", "mog_eval"):
        mog = {"oracle_run": str(latest("exp200_oracle_1m").relative_to(REPO_ROOT)).replace("\\", "/"),
               "incumbent_arm": "muon"}
        if mode == "mog_tuning":
            base = yaml.safe_load(open(REPO_ROOT / "configs/1m/exp160_baselines.yaml"))["lr"]["muon"][0]
            dump(REPO_ROOT / "configs/1m/exp290_mog_lr_tuning.yaml", {
                "experiment_id": "exp290_mog_lr_tuning", "entry": "experiments/sweep.py", "evidence_class": "measured",
                "description": "LR tuning for the oracle-scheduled MOG arm: same 3-point rule {1/3,1,3} x muon tuned LR, seed 0.",
                "seed": 0, "arms_file": "configs/arms.yaml", "arms": ["mog_oracle"],
                "lr": {"mog_oracle": [round(base * f, 8) for f in (1 / 3, 1, 3)]}, "seeds": [0],
                "mog_oracle": mog, "train": TRAIN_1M})
        else:
            f, _ = sweep_summary(latest("exp290_mog_lr_tuning"))
            c = {lr: d[0][0] for (arm, lr), d in f.items() if np.isfinite(d[0][0])}
            dump(REPO_ROOT / "configs/1m/exp300_mog.yaml", {
                "experiment_id": "exp300_mog_1m", "entry": "experiments/sweep.py", "evidence_class": "measured",
                "description": "Oracle-scheduled MOG arm at its tuned LR, evaluation seeds 1-3 (H1 test vs exp160).",
                "seed": 1, "arms_file": "configs/arms.yaml", "arms": ["mog_oracle"],
                "lr": {"mog_oracle": [min(c, key=c.get)]}, "seeds": [1, 2, 3], "mog_oracle": mog, "train": TRAIN_1M})


if __name__ == "__main__":
    main(sys.argv[1])
