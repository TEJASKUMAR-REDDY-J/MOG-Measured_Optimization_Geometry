# EXPERIMENT LOG

Append-only. Every substantive experiment gets an entry here *before* it is run (pre-registration) and is updated with results afterwards. Planned experiment IDs are listed in `experiments/registry.yaml`.

Entry template:

```
## <ID> — <title>
- Date (pre-registered / run):
- Stage / evidence class:
- Hypothesis:
- Config: configs/...  | Run dir(s): results/raw/...
- Baseline:
- Metric / falsification criterion:
- Result (raw):
- Seed variation:
- Interpretation:
- Hypothesis status change:
- Next action / gate decision:
```

---

## infra-000: Initial infrastructure sanity checks
- Date: 2026-09-23
- Stage / evidence class: 0 / unit tests plus a `smoke` pipeline run. This is not an experiment.
- Hypothesis: none. Checks correctness of the implementation only.
- What ran:
  - `python -m pytest -q`: 89 passed. The suite covers LMO and duality identities, C constants (sup property and attainment), special-case coincidences, the partial-whitening identity, NS approximation bounds, a bit-exact Muon equivalence against the reference code, the Cauchy–Schwarz bound, the effective-dimension identities, the PDF §4.3 closed-form columns, secant properties on quadratics, and runner determinism.
  - `exp000_smoke` (64×64, 2 seeds) wrote to `results/scratch/`, which is gitignored and not cited.
- Observations, to be checked by Exp 0 and not claimed as results:
  - The closed-form r_eff and full-spectral columns of PDF §4.3 reproduce to printed precision (a unit test).
  - The smoke run suggests the NS-based nuclear-norm proxy is biased low, by roughly 14–31% depending on the spectrum. Its direction cosine to the exact $UV^\top$ is ≥0.98. Exp 0 will measure this at 512×512.
- Gate decision: infrastructure is ready for Exp 0, pending approval of the experimental phase.

---

## exp000_synthetic_gradient_fit — Synthetic reproduction of PDF §4.3 / Figs 1–2
- Date: 2026-09-23 (pre-registered in the infrastructure commit; first run 2026-09-23)
- Stage / evidence class: 0 / synthetic
- Hypothesis: none of H1–H9. Checks that the PDF's synthetic numbers are computed correctly, and measures how accurate the Newton–Schulz (NS) nuclear-norm proxy is.
- Config: `configs/synthetic/exp000_gradient_fit.yaml` (5 seeds, 512×512, β ∈ 0..1.5)
- Pass criterion (declared now, before the canonical re-run): the closed-form columns match to printed precision, and each Monte Carlo column is within 2 seed-std of the PDF value.
- Note on the first run (`results/raw/exp000_synthetic_gradient_fit/20260923T103502143861Z`): `git_dirty: true`. The cause was a bug: `environment_info()` counted the run's own untracked output directory. The fix excludes `results/`. The code had not changed since commit d0e368a, but the run is re-done cleanly and the first run is kept as superseded.

## exp010_tiny_optimizer_sanity — Stage 2 pilot
- Date: 2026-09-23 (pre-registered)
- Stage / evidence class: 2 / pilot. No hypothesis is tested.
- Purpose: check that all 14 arms train stably somewhere on a 5-point LR grid (2 seeds, ~0.11M params, 300 steps), and locate the grid centre for the 1M tuning sweep.
- GO criterion: every arm has at least one grid LR where both seeds finish without divergence and end below the unigram entropy of the data (~3.3 nats). An arm whose best LR sits at the grid edge is flagged, and its 1M grid is extended in that direction.

## exp011_tiny_atlas_dryrun / exp012_tiny_oracle_dryrun — Stage 2 pipeline pilots
- Date: 2026-09-23 (pre-registered)
- Stage / evidence class: 2 / pilot. Not evidence for H1 or H2.
- Purpose: validate the atlas instrumentation and the oracle + proxy pipeline end to end on the tiny model, and estimate the 1M cost.
- GO criterion: every metric is finite, the oracle finishes, and the per-rollout cost is known.
