# exp012_tiny_oracle_dryrun

- status: ok
- evidence class: pilot
- description: Oracle rollouts + proxy scoring at the exp011 checkpoints (~0.11M model).
- git commit: 365de926bfbf2bde6445e56b4a2c8e6da0817074 (dirty: True)
- wall time: 222.753 s

Reproduce:

```bash
python -m scripts.run_experiment --config configs/tiny/exp012_oracle_dryrun.yaml
```

Machine-readable summary: `results.json`; per-row data: `metrics.csv`.
