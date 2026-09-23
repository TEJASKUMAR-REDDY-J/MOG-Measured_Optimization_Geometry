# exp011_tiny_atlas_dryrun

- status: ok
- evidence class: pilot
- description: Atlas instrumentation dry run on the ~0.11M model (Muon hybrid incumbent + AdamW).
- git commit: 365de926bfbf2bde6445e56b4a2c8e6da0817074 (dirty: True)
- wall time: 30.362 s

Reproduce:

```bash
python -m scripts.run_experiment --config configs/tiny/exp011_atlas_dryrun.yaml
```

Machine-readable summary: `results.json`; per-row data: `metrics.csv`.
