# exp440_h5_vocab

- status: ok
- evidence class: measured
- description: H5: Adam (muon arm) vs per-token rows (duality arm) on embed/pos/head with GPT-2 BPE (~11.7k types). Tune on seed 0, evaluate seeds 1-3, loss by token-frequency decile.
- git commit: cdd26d7af15ddf195ce72f210ac450f88b98b4e8 (dirty: False)
- wall time: 2113.617 s

Reproduce:

```bash
python -m scripts.run_experiment --config configs/redesign/exp440_h5_vocab.yaml
```

Machine-readable summary: `results.json`; per-row data: `metrics.csv`.
