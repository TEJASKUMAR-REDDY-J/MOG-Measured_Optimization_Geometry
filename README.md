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

Stages 0–3 are complete (0.82M-parameter transformer on CPU). Full tables are in [docs/RESULTS.md](docs/RESULTS.md) and the reasoning is in [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md).

| Question | Answer at this scale |
|---|---|
| Do the PDF's synthetic numbers reproduce? | **Yes.** Exp 0 passes, with every column within seed spread. |
| Which hidden-weight geometry wins? | **Spectral (Muon family), by about 0.35 nats over AdamW.** NorMuon, Muon and the duality assignment are tied. |
| H2: does the PDF's free proxy E_c predict the per-block oracle? | **No, FALSIFIED.** Median ρ −0.33, and top-1 agreement is at chance. |
| H1: does an oracle per-block assignment beat the best tuned optimizer? | **Not supported.** +0.015 [−0.001, +0.031] vs NorMuon. |
| Does momentum become low-rank over training (the PDF's premise for H2)? | **No.** r_eff rises. |
| Is there a proxy that does work? | An *own-direction* secant (ρ 0.76, top-1 48%) does, but it costs one extra backward per candidate. It is exploratory and needs its own pre-registration. |

Gate: **NO-GO** for scaling the PDF's selector. The next step is a redesign (EXPERIMENT_LOG, "Gate decision").

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

Run Exp 0 itself:

```bash
python -m scripts.run_experiment --config configs/synthetic/exp000_gradient_fit.yaml
```

Run the full Stage 3 chain (several hours on CPU):

```bash
bash scripts/run_stage3.sh
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
mog/geometry/    base (Geometry, Euclidean), elementwise, rowwise, grouped, spectral (full/grouped/per-expert, alpha)
mog/optim/       geometric (any geometry + momentum), blockwise (per-block rules, switchable),
                 adamw, muon, normuon, mog (online selector: gated on H2)
mog/selection/   metrics, curvature (secant), atlas (profiler metrics), oracle, hysteresis, selector (gated)
mog/models/      tiny_transformer, tiny_mlp, model_factory
mog/training/    trainer, evaluation, logging
mog/utils/       reproducibility (seed, env, configs, run dirs), config, seed, profiling
mog/data.py      char-level Tiny Shakespeare with a seekable batch stream (data not committed)
configs/         synthetic/, tiny/, 1m/, 5m/, larger/, arms.yaml (optimizer arms)
experiments/     00_synthetic ... 06_scaling, sweep.py (generic arm x LR x seed), registry.yaml
scripts/         run_experiment (canonical runner), aggregate_results, generate_plots,
                 profile_overhead, make_stage3_configs, run_stage3.sh
tests/           geometry, muon equivalence, blockwise rules, metrics, curvature, hysteresis, reproducibility
results/         raw/<exp_id>/<timestamp>/ per run (never overwritten), processed/, figures/
reports/         results page (private, work in progress): template, builders, generated data
docs/            plan, theory, hypotheses, protocol, log, results, roadmap, literature,
                 FRAMEWORK (package design), PROBLEM_STATEMENT_CROSS_DOMAIN (spin-off project)
notebooks/       exploration only
```

The dataset is downloaded on first use to `data/tinyshakespeare.txt`, which is gitignored and SHA-256 checked; see `mog/data.py` for the URL.

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

[RESEARCH_PLAN](docs/RESEARCH_PLAN.md) · [FRAMEWORK](docs/FRAMEWORK.md) · [PROBLEM_STATEMENT_CROSS_DOMAIN](docs/PROBLEM_STATEMENT_CROSS_DOMAIN.md) · [THEORY](docs/THEORY.md) · [HYPOTHESES](docs/HYPOTHESES.md) · [EXPERIMENT_PROTOCOL](docs/EXPERIMENT_PROTOCOL.md) · [EXPERIMENT_LOG](docs/EXPERIMENT_LOG.md) · [RESULTS](docs/RESULTS.md) · [ROADMAP](docs/ROADMAP.md) · [LITERATURE](docs/LITERATURE.md)

## License

MIT (see `LICENSE`).
