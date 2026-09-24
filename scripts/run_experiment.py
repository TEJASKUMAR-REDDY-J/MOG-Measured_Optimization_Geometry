"""Canonical entry point for every experiment.

    python -m scripts.run_experiment --config configs/synthetic/exp000_gradient_fit.yaml

Creates results/raw/<experiment_id>/<timestamp>/ containing:
    config.yaml, environment.json, stdout.log, results.json, README.md,
    plus whatever the entry writes (metrics.csv, figures/).
The entry is a Python file named by `entry:` in the config that defines
    main(cfg: dict, run_dir: Path) -> dict   (summary merged into results.json)
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import shlex
import sys
import time
import traceback
from pathlib import Path

import yaml

from mog.utils.reproducibility import REPO_ROOT, environment_info, load_config, new_run_dir, set_seed, write_json


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()

    def close(self):  # logging handlers (e.g. absl via JAX) close their stream at exit; the tee owns nothing
        pass


def _load_entry(path: Path):
    spec = importlib.util.spec_from_file_location(f"mog_entry_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(config_path: str | Path, out_root: str | Path) -> Path:
    cfg = load_config(config_path)
    run_dir = new_run_dir(out_root, cfg["experiment_id"])
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    env = environment_info()
    write_json(run_dir / "environment.json", env)
    command = "python -m scripts.run_experiment " + " ".join(shlex.quote(a) for a in sys.argv[1:])

    set_seed(cfg["seed"])
    entry = _load_entry(REPO_ROOT / cfg["entry"])
    t0 = time.time()
    status, summary, error = "ok", {}, None
    with open(run_dir / "stdout.log", "w", encoding="utf-8") as log, \
            contextlib.redirect_stdout(_Tee(sys.stdout, log)), contextlib.redirect_stderr(_Tee(sys.stderr, log)):
        try:
            summary = entry.main(cfg, run_dir) or {}
        except Exception:
            status, error = "error", traceback.format_exc()
            print(error, file=sys.stderr)
    results = {
        "experiment_id": cfg["experiment_id"],
        "status": status,
        "evidence_class": cfg.get("evidence_class", "unspecified"),
        "seed": cfg["seed"],
        "wall_time_s": round(time.time() - t0, 3),
        "command": command,
        "git_commit": env["git_commit"],
        "git_dirty": env["git_dirty"],
        "summary": summary,
        "error": error,
    }
    write_json(run_dir / "results.json", results)
    (run_dir / "README.md").write_text(
        f"# {cfg['experiment_id']}\n\n"
        f"- status: {status}\n- evidence class: {results['evidence_class']}\n"
        f"- description: {cfg.get('description', '').strip()}\n"
        f"- git commit: {env['git_commit']} (dirty: {env['git_dirty']})\n"
        f"- wall time: {results['wall_time_s']} s\n\n"
        f"Reproduce:\n\n```bash\n{command}\n```\n\n"
        "Machine-readable summary: `results.json`; per-row data: `metrics.csv`.\n",
        encoding="utf-8",
    )
    print(f"[run_experiment] {status}: {run_dir}")
    if status != "ok":
        raise SystemExit(1)
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "raw"),
                    help="output root (use results/scratch for smoke runs)")
    args = ap.parse_args()
    run(args.config, args.out)


if __name__ == "__main__":
    main()
