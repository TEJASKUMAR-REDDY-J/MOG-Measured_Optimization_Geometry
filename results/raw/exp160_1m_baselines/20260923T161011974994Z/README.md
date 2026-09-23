# exp160_1m_baselines

- status: ok
- evidence class: measured
- description: All arms at their exp150/151-tuned LR, evaluation seeds 1-3 (seed 0 reserved for tuning/oracle).
- git commit: fc0beb175e8d51c5b57e39b6a6303efc7a9bc6d4 (dirty: False)
- wall time: 6789.877 s

Reproduce:

```bash
python -m scripts.run_experiment --config configs/1m/exp160_baselines.yaml
```

Machine-readable summary: `results.json`; per-row data: `metrics.csv`.
