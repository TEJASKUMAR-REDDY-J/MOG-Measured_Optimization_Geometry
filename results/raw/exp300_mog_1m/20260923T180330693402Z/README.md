# exp300_mog_1m

- status: ok
- evidence class: measured
- description: Oracle-scheduled MOG arm at its tuned LR, evaluation seeds 1-3 (H1 test vs exp160).
- git commit: fc0beb175e8d51c5b57e39b6a6303efc7a9bc6d4 (dirty: False)
- wall time: 451.025 s

Reproduce:

```bash
python -m scripts.run_experiment --config configs/1m/exp300_mog.yaml
```

Machine-readable summary: `results.json`; per-row data: `metrics.csv`.
