# exp410_h2prime_oracle

- status: ok
- evidence class: measured
- description: H2' pre-registered test of the own-direction proxy: same oracle as exp200 on fresh checkpoints (seed 4) and fresh data streams.
- git commit: cdd26d7af15ddf195ce72f210ac450f88b98b4e8 (dirty: False)
- wall time: 4335.291 s

Reproduce:

```bash
python -m scripts.run_experiment --config configs/redesign/exp410_h2prime_oracle.yaml
```

Machine-readable summary: `results.json`; per-row data: `metrics.csv`.
