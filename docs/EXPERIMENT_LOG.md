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
