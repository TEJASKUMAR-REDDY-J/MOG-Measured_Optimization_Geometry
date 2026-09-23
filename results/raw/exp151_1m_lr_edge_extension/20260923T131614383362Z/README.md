# exp151_1m_lr_edge_extension

- status: ok
- evidence class: measured
- description: Pre-registered edge rule: one extra LR beyond the edge for arms whose exp150 best sat at a grid edge.
- git commit: 70eb1ab2b91282d19781e52868fc0fc707afe667 (dirty: False)
- wall time: 1252.054 s

Reproduce:

```bash
python -m scripts.run_experiment --config configs/1m/exp151_lr_edge_extension.yaml
```

Machine-readable summary: `results.json`; per-row data: `metrics.csv`.
