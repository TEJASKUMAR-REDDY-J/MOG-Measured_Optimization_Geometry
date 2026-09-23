# exp010_tiny_optimizer_sanity

- status: ok
- evidence class: pilot
- description: Every arm in configs/arms.yaml on a ~0.11M-param char transformer, 5-point LR grid per family, 2 seeds, 300 steps. Checks that every rule trains stably somewhere on its grid and locates the grid centre for the 1M tuning sweep.
- git commit: 365de926bfbf2bde6445e56b4a2c8e6da0817074 (dirty: True)
- wall time: 1358.288 s

Reproduce:

```bash
python -m scripts.run_experiment --config configs/tiny/exp010_optimizer_sanity.yaml
```

Machine-readable summary: `results.json`; per-row data: `metrics.csv`.
