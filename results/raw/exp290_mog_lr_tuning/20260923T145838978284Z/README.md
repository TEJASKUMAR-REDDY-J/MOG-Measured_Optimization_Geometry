# exp290_mog_lr_tuning

- status: ok
- evidence class: measured
- description: LR tuning for the oracle-scheduled MOG arm: same 3-point rule {1/3,1,3} x muon tuned LR, seed 0.
- git commit: b4b7a396009a05b578de47dcf0bdb1ef568412ac (dirty: False)
- wall time: 515.103 s

Reproduce:

```bash
python -m scripts.run_experiment --config configs/1m/exp290_mog_lr_tuning.yaml
```

Machine-readable summary: `results.json`; per-row data: `metrics.csv`.
