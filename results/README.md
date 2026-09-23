# results/

| Directory | Contents |
|---|---|
| `raw/<experiment_id>/<UTC timestamp>/` | One directory per run, written by `scripts/run_experiment.py` and never overwritten. Contains `config.yaml`, `environment.json`, `stdout.log`, `results.json`, `README.md`, `metrics.csv` and `figures/`. |
| `processed/` | Cross-run tables built by `python -m scripts.aggregate_results`. |
| `figures/` | Cross-run figures for the docs. |
| `scratch/` | Smoke runs. Gitignored and never cited. |

Checkpoints and datasets never go here (see `.gitignore`).
