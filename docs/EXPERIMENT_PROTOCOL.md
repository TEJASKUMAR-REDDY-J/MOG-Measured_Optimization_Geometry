# EXPERIMENT PROTOCOL

## 1. Evidence classes

Every run declares an `evidence_class` in its config, and the class is copied into `results.json`.

| class | meaning | may appear in RESULTS.md? |
|---|---|---|
| `smoke` | Pipeline check. Written to `results/scratch/` (gitignored). | never |
| `synthetic` | Synthetic matrices or closed-form evaluation. No training. | yes, in the "synthetic/theory checks" section only |
| `pilot` | Tiny-scale (Stage 2) debugging run | yes, labeled pilot, no scientific claims |
| `measured` | A pre-registered experiment at Stage ≥3 | yes |

## 2. Running

```bash
python -m scripts.run_experiment --config configs/<stage>/<exp>.yaml
```

The runner:
- creates `results/raw/<experiment_id>/<UTC timestamp>/` and never reuses a directory
- writes `config.yaml`, `environment.json` (versions, platform, CPU/GPU, git commit and dirty flag, argv) and `stdout.log` (stdout and stderr through a tee)
- sets the seed and calls the entry's `main(cfg, run_dir)`
- writes `results.json` (status, evidence class, seed, wall time, exact command, git commit, summary) and a human-readable `README.md`

Entries write `metrics.csv` and `figures/`.

A run is **canonical** only if `git_dirty` is `false`. Commit the code first, then run.

To regenerate the cross-run index:

```bash
python -m scripts.aggregate_results
```

It writes `results/processed/runs_index.csv`.

## 3. Pre-registration

Before running any `measured` experiment, add an EXPERIMENT_LOG entry that states:
- the hypothesis
- the metric
- the falsification criterion
- the seeds
- the checkpoints and blocks to analyze
- the statistical test

Changing the analysis after seeing results requires a new, dated entry that explains why. The original entry is not edited.

## 4. Budgets and baselines

- **Matched tokens** for every comparison. Also report **matched wall-clock** once NS overhead matters.
- **Equal tuning budget.** Every arm gets the same number of LR (and other) sweep points on the same grid type. Record the number of tuned scalars per arm.
- Baselines, eventually: SGD+momentum (where useful), AdamW, Muon (hidden) + AdamW (rest), NorMuon, the fixed folklore hybrid, and MOG.
- Report Pareto curves (loss vs tokens and vs wall-clock), not single points, from Stage 3 onward.

## 5. Seeds and statistics

- Pilot: ≥2 seeds. Any claim: ≥3 seeds. Headline claims: ≥5 seeds.
- Report mean ± std and the paired difference against the baseline with a 95% t-interval.
- "Within noise" means the 95% CI of the paired difference includes 0.
- The oracle and proxy analyses report the per-block Spearman ρ distribution (median, IQR) and top-1 agreement against a chance baseline.

## 6. Oracle procedure (Exp 2, specification)

At pre-declared checkpoints (e.g. 6 evenly spaced), take each block and each candidate geometry:
- restore the checkpoint (weights, momentum and RNG)
- apply the candidate to **that block only** for h steps, leaving everything else unchanged, on a fixed data stream shared across candidates
- record the realized loss reduction on a fixed held-out batch set

The rollout-noise floor is estimated by repeating the incumbent rollout with different data orders. Each candidate's step size must be chosen by a declared rule, either Prop. 1's $t^\star$ or a small per-candidate sweep, and that rule is logged.

## 7. Integrity rules

- Never overwrite or delete a run directory. Superseded runs stay, and the log says why they were superseded.
- Never cherry-pick seeds, blocks, checkpoints or model sizes.
- No results only in notebooks. Canonical numbers come from `results/raw/*/results.json` or `metrics.csv`.
- Checkpoints, datasets and caches are not committed (`.gitignore`).
