"""Block statistics at oracle checkpoints (proposal H1 predictors). One row per
(source, checkpoint, block); definitions in mog/selection/block_probe.py.

Config: sources {name: spec} (as in oracle_tasks.py), checkpoints, K, probe_seed, workers.
"""

from __future__ import annotations

import csv
import importlib.util
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from mog.training.jobs import _load, blocks_meta, run_probe
from mog.utils.reproducibility import REPO_ROOT

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
_spec = importlib.util.spec_from_file_location("oracle_tasks", Path(__file__).with_name("oracle_tasks.py"))
_ot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ot)


def main(cfg: dict, run_dir: Path) -> dict:
    jobs = []
    for sname, spec in cfg["sources"].items():
        src = _ot.resolve_source(spec)
        for s in cfg["checkpoints"]:
            jobs.append((sname, s, _ot.ckpt_file(src, s)))
    with ProcessPoolExecutor(cfg.get("workers", 4)) as pool:
        futs = [pool.submit(run_probe, p, cfg["K"], cfg["probe_seed"]) for *_, p in jobs]
        rows = []
        for (sname, s, path), f in zip(jobs, futs):
            meta = blocks_meta(_load(path))
            for block, st in f.result().items():
                rows.append({"source": sname, "ckpt": s, "block": block, "role": meta[block]["role"], "kind": meta[block]["kind"]} | st)
            print(f"{sname} step {s}: {len(rows)} rows", flush=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("source", "ckpt", "block", "role", "kind"), k))
    with open(run_dir / "block_stats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return {"n_rows": len(rows)}
