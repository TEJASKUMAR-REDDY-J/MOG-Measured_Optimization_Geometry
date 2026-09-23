# ROADMAP

## Done
- [x] Repository skeleton, packaging, docs, and literature (all IDs verified)
- [x] Geometry interface and five geometry families. `GeometricOptimizer` is bit-exact with Muon.
- [x] `BlockOptimizer`: 12 per-block rules, switchable mid-run. Tested against torch Adam and reference Lion.
- [x] Metrics, secant curvature, atlas profiler metrics, hysteresis rule
- [x] Runner, reproducibility, results index, report builders
- [x] Exp 0 synthetic (PASS)
- [x] Stage 2: exp010 sweep, exp011/012 pilots (GO)
- [x] Stage 3: exp150/151 → exp100 → exp200 → exp290 → exp160 → exp300 (gate: NO-GO for scaling; redesign)

## Next
- [ ] Redesign pre-registrations: H2' own-direction proxy, joint/longer-horizon oracle, H4 random-partition control
- [ ] Results page (`reports/`, private, UI paused)
- [ ] Framework extraction (docs/FRAMEWORK.md): generic block discovery; atlas and oracle behind a library API
- [ ] Stage 1 synthetic extensions (non-square shapes, sizes, rank sweeps)

## Gated modules (present, raise NotImplementedError)
| Module | Gate |
|---|---|
| `mog/optim/mog.py`, `mog/selection/selector.py` | NO-GO: Exp 2 falsified H2. Revisit only if a redesigned proxy (H2') passes |

## Structure notes
- `mog/geometry/grouped.py` defines `GroupedSpectral` on top of `Spectral(k=...)`. Whole-matrix spectral is the k=m case of grouped, so the math lives in one place.
- `mog/utils/{config,seed}.py` re-export from `reproducibility.py`, which holds the shared implementation.
- `mog/selection/oracle.py` holds the oracle scoring utilities. The rollout driver stays in `experiments/03_oracle/oracle.py` until Stage 3 finishes executing it, and moves into the library afterwards.
- There is no `uv.lock` because `uv` is not installed here. Exact package versions are captured per run in `environment.json`.
- Notebooks are created only when used for exploration (`notebooks/README.md`).
