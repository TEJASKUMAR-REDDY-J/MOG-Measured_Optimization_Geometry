# exp430_h4_randpart_tuning

- status: ok
- evidence class: measured
- description: H4 control: random-partition grouped spectral on attention (head-sized groups), same 3-point LR rule, seed 0.
- git commit: 2c9b45800dfd734048e65463d5c860271252bffb (dirty: False)
- wall time: 396.222 s

Reproduce:

```bash
python -m scripts.run_experiment --config configs/redesign/exp430_h4_randpart_tuning.yaml
```

Machine-readable summary: `results.json`; per-row data: `metrics.csv`.
