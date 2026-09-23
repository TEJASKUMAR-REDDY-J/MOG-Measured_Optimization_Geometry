# exp100_atlas_1m

- status: ok
- evidence class: measured
- description: Geometry atlas on the 0.82M model: muon (incumbent, writes oracle checkpoints) and adamw, seed 0, tuned LRs.
- git commit: 2abf18a81ecb50b7109a59c0115eb5e24551bd6c (dirty: False)
- wall time: 366.037 s

Reproduce:

```bash
python -m scripts.run_experiment --config configs/1m/exp100_atlas.yaml
```

Machine-readable summary: `results.json`; per-row data: `metrics.csv`.
