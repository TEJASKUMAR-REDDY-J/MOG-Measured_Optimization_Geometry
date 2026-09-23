# exp000_synthetic_gradient_fit

- status: ok
- evidence class: synthetic
- description: Reproduce PDF §4.3 table and Fig. 1b/1c/1d/2a on 512x512 Haar-random power-law momentum matrices; measure Newton-Schulz dual-norm fidelity.
- git commit: d0e368a9b13f9bcce66c4d992e344c914f111057 (dirty: True)
- wall time: 466.328 s

Reproduce:

```bash
python -m scripts.run_experiment --config configs/synthetic/exp000_gradient_fit.yaml
```

Machine-readable summary: `results.json`; per-row data: `metrics.csv`.
