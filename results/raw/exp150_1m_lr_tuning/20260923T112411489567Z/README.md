# exp150_1m_lr_tuning

- status: ok
- evidence class: measured
- description: Stage-3 LR tuning: 3-point grid {1/3,1,3} x exp010 tiny-best LR per arm, seed 0 only.
- git commit: 3078e3093031a5bc41548e1d31ce64ea89e0dd11 (dirty: False)
- wall time: 6708.297 s

Reproduce:

```bash
python -m scripts.run_experiment --config configs/1m/exp150_lr_tuning.yaml
```

Machine-readable summary: `results.json`; per-row data: `metrics.csv`.
