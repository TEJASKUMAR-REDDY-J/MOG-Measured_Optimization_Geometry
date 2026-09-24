# exp420_additivity

- status: ok
- evidence class: measured
- description: Additivity and horizon of the exp200 stable switches: each alone vs all jointly, h=20 and h=100, fresh data stream 779.
- git commit: cdd26d7af15ddf195ce72f210ac450f88b98b4e8 (dirty: False)
- wall time: 1262.138 s

Reproduce:

```bash
python -m scripts.run_experiment --config configs/redesign/exp420_additivity.yaml
```

Machine-readable summary: `results.json`; per-row data: `metrics.csv`.
