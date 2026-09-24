# exp400_h2prime_ckpts

- status: ok
- evidence class: measured
- description: Fresh incumbent (muon hybrid, tuned lr 0.01, NEW seed 4) writing checkpoints 100/300/500 for the H2' oracle. atlas_every is set beyond the run length, so no atlas measurements are taken.
- git commit: cdd26d7af15ddf195ce72f210ac450f88b98b4e8 (dirty: False)
- wall time: 431.993 s

Reproduce:

```bash
python -m scripts.run_experiment --config configs/redesign/exp400_h2prime_ckpts.yaml
```

Machine-readable summary: `results.json`; per-row data: `metrics.csv`.
