# MOG: Measured Optimization Geometry

MOG asks whether the **update geometry** of each neural-network parameter block should be **selected from measurement**, rather than fixed by convention (Muon on 2-D hidden weights, AdamW on everything else).

The idea: AdamW and Muon are steepest descent under different norms. Elementwise ℓ∞ corresponds to sign/Adam-like updates, and the whole-matrix spectral norm corresponds to Muon. Between those corners lies a lattice of geometries: rows, column groups, k-row spectral groups, per-head blocks, per-expert blocks, and partial whitening. The PDF proposes choosing among them per block by a measured descent efficiency, $E_c=\theta_c/\hat L_c$ (gradient-fit ÷ secant curvature).

The primary specification is the research document `granular_optimizer_geometry.pdf`, summarized in [docs/THEORY.md](docs/THEORY.md). That document contains **no experimental results**. Its numbers are closed-form or come from synthetic spectra.

## Research question

1. Is the optimal geometry heterogeneous across blocks?
2. Can it be identified online from quantities already computed during training?
3. Does acting on it improve the loss/compute frontier over the best tuned homogeneous optimizer at matched budget?

The ladder is theory → measurement → oracle → proxy validation → online selection → scaling. See [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md).

## Current status (2026-09-23)

- **Stage 0 infrastructure only. No training has been run. No hypothesis has been tested.**
- The geometry interface and five geometry families are implemented and unit-tested (89 tests).
- The generic geometry optimizer matches reference Muon bit-for-bit.
- The Exp 0 synthetic framework is implemented and smoke-tested, but has not yet been run at full size.

### Verified findings

These are unit-test or closed-form checks. None of them is a training result.

- The PDF §4.3 closed-form columns reproduce: r_eff = 512 / 281.65 / 28.28 / 5.30, and full-spectral θ/C = 1.000 / 0.5501 / 0.0552 / 0.0104.
- The LMO/duality identities, the C constants, and the Cauchy–Schwarz bound θ_c/C_c ≤ 1 hold for every implemented geometry.
- The secant curvature estimate upper-bounds the directional Rayleigh quotient. It is not Prop. 1's sup-curvature (THEORY.md §5).
- Reference checks: all 32 arXiv identifiers resolve. Two PDF claims are overstated or unverified: the Gluon venue and the boundary-layer attribution (THEORY.md §11).

### Current unknowns

- Whether Muon's approximate Newton–Schulz makes the "free" nuclear-norm proxy materially biased. The smoke run hints yes; Exp 0 will quantify it.
- Whether real momenta sit in the low-rank regime where §4.3 predicts large geometry differences.
- Whether a secant estimate from consecutive minibatches measures curvature or gradient noise.
- H1–H9: all NOT TESTED ([docs/HYPOTHESES.md](docs/HYPOTHESES.md)).

## Installation

Requires Python ≥3.10, PyTorch ≥2.2, NumPy, PyYAML and Matplotlib (pytest for the tests). Current development runs on CPU (torch 2.8.0+cpu, Python 3.12).

```bash
pip install -e ".[dev]"
```

Installing is optional: run everything from the repo root with `python -m ...`.

## Quick start

Run the test suite:

```bash
python -m pytest -q
```

Run the Exp 0 pipeline smoke check (writes to the gitignored `results/scratch/`):

```bash
python -m scripts.run_experiment --config configs/synthetic/exp000_smoke.yaml --out results/scratch
```

Run Exp 0 itself (not yet run; awaiting approval):

```bash
python -m scripts.run_experiment --config configs/synthetic/exp000_gradient_fit.yaml
```

Use any geometry as an optimizer:

```python
from mog.geometry import Spectral, RowNorm
from mog.optim import GeometricOptimizer, muon_scale

opt = GeometricOptimizer(hidden_params, Spectral(), lr=0.02, scale=muon_scale)  # = Muon
opt = GeometricOptimizer(embedding_params, RowNorm("rows"), lr=0.01)            # per-token rows
```

## Layout

```
mog/geometry/    Geometry interface: Euclidean, Elementwise, RowNorm, Spectral (full/grouped/per-expert, α)
mog/optim/       GeometricOptimizer (any geometry + momentum; Muon-equivalent config)
mog/selection/   gradient-fit / effective dimensions / E_c; secant curvature
mog/utils/       seeds, environment capture, configs, non-overwriting run dirs
configs/         one YAML per experiment
experiments/     experiment entry points + registry.yaml
scripts/         run_experiment.py (canonical runner), aggregate_results.py
tests/           unit / math / reproducibility tests
results/         raw/<exp_id>/<timestamp>/ per run; processed/; figures/
docs/            plan, theory, hypotheses, protocol, log, results, roadmap, literature
```

## Experiment roadmap

| Stage | Scale | Content |
|---|---|---|
| 0 | none | Unit tests plus Exp 0 synthetic |
| 1 | synthetic | Synthetic extensions |
| 2 | ≤1M | Pipeline pilots |
| 3 | ~1M | Atlas, oracle, proxy |
| 4 | ~5M | Informative repeats, online selection if justified |
| 5 | 20–50M | Scaling |
| 6 | 124M | Only with evidence |

Each stage has an explicit GO / NO-GO gate ([docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md) §4). Deferred modules are listed in [docs/ROADMAP.md](docs/ROADMAP.md).

## Documentation

[RESEARCH_PLAN](docs/RESEARCH_PLAN.md) · [THEORY](docs/THEORY.md) · [HYPOTHESES](docs/HYPOTHESES.md) · [EXPERIMENT_PROTOCOL](docs/EXPERIMENT_PROTOCOL.md) · [EXPERIMENT_LOG](docs/EXPERIMENT_LOG.md) · [RESULTS](docs/RESULTS.md) · [ROADMAP](docs/ROADMAP.md) · [LITERATURE](docs/LITERATURE.md)

## License

MIT (see `LICENSE`).
