# exp200_oracle_1m

- status: ok
- evidence class: measured
- description: Oracle geometry per block (single-block switch, h=20 steps, constant checkpoint LR) at steps 100/300/500 of the exp100 muon seed-0 run; proxy validation (H1, H2). Pre-registered in EXPERIMENT_LOG.
- git commit: 2abf18a81ecb50b7109a59c0115eb5e24551bd6c (dirty: False)
- wall time: 4489.065 s

Reproduce:

```bash
python -m scripts.run_experiment --config configs/1m/exp200_oracle.yaml
```

Machine-readable summary: `results.json`; per-row data: `metrics.csv`.
