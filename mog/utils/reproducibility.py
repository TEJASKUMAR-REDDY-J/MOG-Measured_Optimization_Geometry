"""Seeds, environment capture, config loading, and non-overwriting run directories."""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("experiment_id", "entry", "seed"):
        if key not in cfg:
            raise KeyError(f"config {path} missing required key {key!r}")
    return cfg


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def environment_info() -> dict:
    # results/ is excluded: the run's own output dir exists before this is called
    status = _git("status", "--porcelain", "--", ".", ":(exclude)results")
    return {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "argv": sys.argv,
    }


def new_run_dir(root: str | Path, experiment_id: str) -> Path:
    """results/raw/<experiment_id>/<UTC timestamp>; never reuses an existing directory."""
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    base = Path(root) / experiment_id
    base.mkdir(parents=True, exist_ok=True)
    for i in range(1000):  # coarse OS clocks (Windows) can repeat a timestamp
        run_dir = base / (stamp if i == 0 else f"{stamp}_{i}")
        try:
            run_dir.mkdir(exist_ok=False)
            return run_dir
        except FileExistsError:
            continue
    raise FileExistsError(f"could not allocate a fresh run dir under {base}")


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
