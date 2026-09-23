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

---

## Stage 3 pre-registration (written 2026-09-23, before any 1M run)

Common setup: GPT with d=128, 4 layers, 4 heads, ctx 128 (0.82M params); char-level Tiny Shakespeare; batch 16×128 = 2048 tokens/step; 600 steps (1.23M tokens, about 1.2 epochs); warmup 60; cosine to 0.1×; weight decay 0; val loss on 8 fixed held-out batches. CPU only.

Seed policy: seed 0 is used only for LR tuning, the atlas and the oracle. Every reported comparison uses seeds 1, 2, 3. This keeps the oracle-derived MOG arm from being evaluated on the run it was derived from.

### exp150_1m_lr_tuning (tuning, not a result)
- Each arm gets 3 LRs = {1/3, 1, 3} × its exp010 tiny-best LR, on seed 0, for 600 steps.
- Edge rule: if the best LR is at an edge of the grid, one more point is added beyond that edge. This is applied the same way to every arm and logged.
- Each arm has exactly one tuned scalar (lr). Betas, eps, the 0.2 RMS target and the NS steps are fixed defaults, not tuned.

### exp160_1m_baselines (measured)
- The 14 arms at their tuned LR, on seeds 1–3.
- Metric: final val loss (mean ± std), with paired differences against `muon` (folklore) and `adamw` and their 95% t-intervals (df=2).
- This is descriptive: optimizer ranking at matched tokens and a matched tuning budget. H4 (symmetry-matched grouping) gets a partial reading from headmuon vs muon. It is only partial because the random-partition control is not run, so H4 cannot be SUPPORTED from this alone. H5 cannot be tested at char level (vocab 65), and duality vs adamw on vocab blocks is reported descriptively only.

### exp100_atlas_1m (measured, descriptive)
- Incumbents: `muon` (checkpoints at steps 100/300/500 for the oracle) and `adamw`, both at tuned LR on seed 0. The atlas is logged every 50 steps.
- Pre-declared readouts:
  - (a) r_eff trend: the fraction of blocks where r_eff_frac falls between the first and last measurement. This is H2's sub-claim.
  - (b) free vs same-batch secant: the median ratio Lfree/Lsame per geometry. A ratio ≫ 1 means consecutive-batch secants are dominated by gradient noise (THEORY §5.5).
  - (c) the distribution of argmax E_c per block kind.

### exp200_oracle_1m (measured; H1 and H2)
- Oracle: at checkpoints 100/300/500 of the `muon` seed-0 run, each 2-D block is switched alone to each candidate rule for h=20 steps at the checkpoint's LR, on data stream 777. The measure is val loss vs an all-incumbent rollout on the same stream. The best candidate is re-run on stream 778.
- H1 evidence at this stage: count the (checkpoint, block) pairs whose best rule differs from the incumbent **and** keeps a positive gain on stream 778.
- H2: take the per-(checkpoint, block) Spearman ρ between each proxy and the oracle gain across candidates.
  - H2 SUPPORTED (PDF criterion): the free proxy has median ρ > 0.7 and its pick recovers ≥70% of the mean oracle-best gain.
  - H2 FALSIFIED: median ρ ≤ 0.3, or top-1 agreement not above chance (one-sided binomial test, p<0.05).
  - Anything in between is INCONCLUSIVE.
  - The other proxies (same, own, fit, lookahead) are reported alongside as diagnostics.

### exp300_mog_1m (measured; H1)
- `mog_oracle` arm: the incumbent `muon` rules, plus the stable oracle switches applied piecewise from checkpoints 100/300/500 (the 100 plan applies from step 0). Its LR is tuned on the same 3-point grid on seed 0, then it is evaluated on seeds 1–3.
- H1 SUPPORTED if mog_oracle beats the best tuned baseline of exp160 with a paired 95% CI excluding 0. FALSIFIED or INCONCLUSIVE if the CI includes 0; note that with 3 seeds only a large effect can be detected.
- `mog_proxy` (online selection with hysteresis) runs **only if** exp200 shows the free or same-batch proxy passes the H2 "not falsified" bar. If it does not run, that is recorded as a NO-GO.

### exp010 note (2026-09-23)
- Run `results/raw/exp010_tiny_optimizer_sanity/20260923T105644*` records `git_dirty: true`. `git diff 365de92 -- mog experiments configs/tiny configs/arms.yaml scripts/run_experiment.py` is **empty**, so the code that ran is exactly commit 365de92. The dirty state came only from documentation edits and new, unused helper scripts that were added while the run was in progress. The run is kept as canonical, and from here on every launch is preceded by a commit.
