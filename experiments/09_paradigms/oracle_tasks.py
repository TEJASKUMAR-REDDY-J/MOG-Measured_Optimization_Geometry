"""Paired unit oracle for any task (generalises experiments/08_shape_scale/oracle_units.py, which
stays frozen for exp505).

From each checkpoint of each source run, switch a UNIT to candidate rule c (every other block
keeps its incumbent rule), train h steps (PPO updates for RL) at the checkpoint's LR on stream
data_seed, and record the task's val loss. gain(c) = L(incumbent) - L(c); paired by construction.

Config:
  sources: {name: <run dir or latest:<experiment_id>/<subdir>>}
  checkpoints: [steps]
  candidates: [rules]
  groups: {unit name: selector}   selector = "role:hidden" | "kind:fc" | "name:<regex>" ; class units
  class_units: {horizons: [...], data_seeds: [...]}
  block_units: {sources: [...], horizons: [...], data_seeds: [...]}   # every non-gain block alone
  contrasts: [[a, b], ...]
  workers: 4
"""

from __future__ import annotations

import csv
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from mog.selection.oracle import unit_contrasts
from mog.training.jobs import _load, blocks_meta, run_rollout
from mog.utils.reproducibility import REPO_ROOT

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")


def resolve_source(spec: str) -> Path:
    if spec.startswith("latest:"):
        exp, _, sub = spec.split(":", 1)[1].partition("/")
        return sorted((REPO_ROOT / "results" / "raw" / exp).iterdir())[-1] / sub
    return REPO_ROOT / spec


def ckpt_file(src: Path, step: int) -> str:
    for ext in (".pt", ".pkl"):
        p = src / "checkpoints" / f"step_{step}{ext}"
        if p.exists():
            return str(p)
    raise FileNotFoundError(src / "checkpoints" / f"step_{step}")


def select(meta: dict, selector: str) -> list[str]:
    field, _, val = selector.partition(":")
    if field == "name":
        return [n for n, m in meta.items() if m["role"] != "gain" and re.search(val, n)]
    return [n for n, m in meta.items() if m[field] == val]


def incumbent_rules(ck) -> dict:
    if "state" in ck:  # RL runner state
        from mog.rl.rules import RULES
        return {k: RULES[int(v)] for k, v in ck["state"]["rule_ids"].items()}
    return ck["opt"]["rules"]


def main(cfg: dict, run_dir: Path) -> dict:
    jobs = []
    for sname, spec in cfg["sources"].items():
        src = resolve_source(spec)
        print(f"source {sname}: {src.relative_to(REPO_ROOT)}")
        for s in cfg["checkpoints"]:
            path = ckpt_file(src, s)
            ck = _load(path)
            meta, rules = blocks_meta(ck), incumbent_rules(ck)
            units = [("class", g, select(meta, sel)) for g, sel in cfg["groups"].items()]
            if sname in cfg["block_units"]["sources"]:
                units += [("block", n, [n]) for n, m in meta.items() if m["role"] != "gain"]
            for utype, uname, members in units:
                u = cfg[f"{utype}_units"]
                for h in u["horizons"]:
                    for ds in u["data_seeds"]:
                        for rule in cfg["candidates"]:
                            if {rules[n] for n in members} == {rule}:
                                continue
                            jobs.append(({"source": sname, "ckpt": s, "unit_type": utype, "unit": uname, "role": meta[members[0]]["role"],
                                          "horizon": h, "data_seed": ds, "rule": rule}, path, {n: rule for n in members}, h, ds))
                        jobs.append(({"source": sname, "ckpt": s, "unit_type": "base", "unit": "-", "role": "-", "horizon": h,
                                      "data_seed": ds, "rule": "incumbent"}, path, {}, h, ds))
    seen, uniq = set(), []
    for j in jobs:
        key = (j[1], tuple(sorted(j[2].items())), j[3], j[4])
        if key not in seen:
            seen.add(key)
            uniq.append(j)
    print(f"{len(uniq)} rollouts", flush=True)
    with ProcessPoolExecutor(cfg.get("workers", 4)) as pool:
        futs = [pool.submit(run_rollout, p, sw, h, ds) for _, p, sw, h, ds in uniq]
        losses = []
        for i, f in enumerate(futs):
            losses.append(f.result())
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(uniq)}", flush=True)
    base = {(m["source"], m["ckpt"], m["horizon"], m["data_seed"]): L for (m, *_), L in zip(uniq, losses) if m["unit_type"] == "base"}
    rows = [m | {"loss": L, "base_loss": base[(m["source"], m["ckpt"], m["horizon"], m["data_seed"])],
                 "gain": base[(m["source"], m["ckpt"], m["horizon"], m["data_seed"])] - L}
            for (m, *_), L in zip(uniq, losses) if m["unit_type"] != "base"]
    with open(run_dir / "oracle_units.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return unit_contrasts(rows, cfg["candidates"], cfg["contrasts"],
                          lambda r: r["unit"] if r["unit_type"] == "class" else f"blocks:{r['role']}")
