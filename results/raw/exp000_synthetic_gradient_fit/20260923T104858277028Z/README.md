# exp000_synthetic_gradient_fit

- status: ok
- evidence class: synthetic
- description: Reproduce PDF §4.3 table and Fig. 1b/1c/1d/2a on 512x512 Haar-random power-law momentum matrices; measure Newton-Schulz dual-norm fidelity.
- git commit: 365de926bfbf2bde6445e56b4a2c8e6da0817074 (dirty: False)
- wall time: 446.862 s

Reproduce:

```bash
python -m scripts.run_experiment --config configs/synthetic/exp000_gradient_fit.yaml
```

Machine-readable summary: `results.json`; per-row data: `metrics.csv`.
