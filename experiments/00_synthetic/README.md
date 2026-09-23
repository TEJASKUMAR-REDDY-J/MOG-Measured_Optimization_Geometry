# Exp 0: synthetic geometry analysis (Stage 0)

**Evidence class: synthetic.** Nothing here is a training measurement.

This experiment independently reproduces PDF §4.3 (the table) and Figs. 1b, 1c, 1d and 2a. It uses 512×512 momentum matrices with Haar-random singular vectors and power-law spectra $\sigma_i=i^{-\beta}$. It also measures something the PDF assumes but does not report: how accurately Muon's Newton–Schulz output recovers $\|B\|_{\rm nuc}$ and $UV^\top$ in float32 and bf16.

```bash
python -m scripts.run_experiment --config configs/synthetic/exp000_gradient_fit.yaml
```

Pipeline smoke check (writes to the gitignored scratch dir):

```bash
python -m scripts.run_experiment --config configs/synthetic/exp000_smoke.yaml --out results/scratch
```

Outputs are written to `results/raw/exp000_synthetic_gradient_fit/<timestamp>/`:

| File | Contents |
|---|---|
| `metrics.csv` | One row per (seed, β, geometry, k, method) |
| `results.json` | Table vs PDF values with abs diffs, grouped means, NS fidelity, closed-form panels |
| `figures/*.png` | Figure analogues, each labeled SYNTHETIC |

Assumptions not stated in the PDF are recorded in the config: the 4096×4096 layer shape for Fig. 1d and the $s^2$ parameterization for Fig. 2a.
