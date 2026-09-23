# ROADMAP

## Done (2026-09-23)
- [x] Repository skeleton, packaging, `.gitignore`, license
- [x] Core docs: plan, theory, hypotheses, protocol, literature (all IDs verified), log, results
- [x] Geometry interface (`lmo`, `norm`, `dual_norm`, `C`, `cost`) with Euclidean, Elementwise, RowNorm (rows/cols) and Spectral (full / k-row / k-col / per-expert batch, α∈[0,1] via SVD, α=0 via NS)
- [x] `GeometricOptimizer`: any geometry as a momentum optimizer. Bit-exact with reference Muon.
- [x] Metrics: θ_c, θ_c/C_c, r_eff, d_eff, m_eff, E_c. Secant curvature.
- [x] Experiment runner, reproducibility utilities, run index
- [x] Exp 0 framework (synthetic) with configs; smoke-tested only

## Next, pending approval
- [ ] Run Exp 0 (`exp000_synthetic_gradient_fit`) and record it in the log and RESULTS
- [ ] Stage 1 extensions: non-square shapes, sizes 64–2048, rank-deficient and condition-number sweeps
- [ ] Choose a dataset (small char-level corpus) and get approval to download it
- [ ] `mog/models/`: tiny MLP and tiny transformer (per-head-aware parameter naming)
- [ ] `mog/training/`: trainer with matched-token budgets, CSV logging and eval
- [ ] Stage 2: exp010 (optimizer sanity) and exp011 (atlas dry run)

## Deferred on purpose (built when their stage is reached)
These are in the requested skeleton but are **not created yet**. Building them before the measurements they depend on would build the adaptive optimizer before the evidence exists (RESEARCH_PLAN §1).

| Planned module | Needed at | Why deferred |
|---|---|---|
| `mog/selection/oracle.py` | Exp 2 | Its design depends on the checkpoint and rollout details learned in Exp 1 |
| `mog/selection/selector.py`, `hysteresis.py` (plus `tests/test_hysteresis.py`) | Exp 3 | Only built if Exp 2 shows the proxy predicts the oracle |
| `mog/optim/mog.py` | Exp 3 | Same |
| `mog/optim/normuon.py` | Stage 3 baselines | Baseline; add when baselines are compared |
| `mog/optim/adamw.py` | — | Not needed: `torch.optim.AdamW` is the baseline |
| `mog/utils/profiling.py`, `scripts/profile_overhead.py` | Stage 2 | Profile after correctness is established |
| `scripts/generate_plots.py` | — | Each experiment writes its own figures, and cross-run plots come later |
| `notebooks/` | any | Exploration only. Canonical results never live there. |
| Geometry-matched weight decay (PDF §5) | Exp 3 | Spec only |
| Global damping γ = 1/λ_max(M̃) (Prop. 3) | Exp 3 / H7 | Spec only |
| Empirical-Bayes shrinkage λ̂ (Prop. 4) | Exp 3 / H3 | Spec only |
| α∈(0,1) via polynomial iteration | Optional candidate | SVD path exists for analysis |

## Structural deviations from the requested skeleton
- `mog/geometry/grouped.py` was merged into `spectral.py`. Whole-matrix spectral is exactly the k=m case of grouped spectral, so a single `Spectral(k=...)` class avoids duplicated code, and the tests check the k=1 → rows and k=m → full limits.
- `mog/utils/{config,seed,reproducibility}.py` were merged into `reproducibility.py`, which is one small module.
- `experiments/01_…06_` directories are created when each experiment is implemented. Until then they are listed in `experiments/registry.yaml`.
- No `uv.lock`, because `uv` is not installed on this machine. Dependencies are declared in `pyproject.toml`, and exact versions are captured per run in `environment.json`.
