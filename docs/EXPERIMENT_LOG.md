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

## Stage 3 progress checkpoint (2026-09-23, chain paused for a machine restart)
- Complete and committed: exp150 and exp151 (LR tuning), exp100 (atlas), exp200 (oracle), exp290 (MOG-oracle LR tuning, best lr 0.01).
- Aborted: `results/raw/exp160_1m_baselines/20260923T150729007364Z` after 19 of 42 runs. It is kept, not deleted, with an ABORTED.txt marker and no results.json, so the report builder ignores it. exp160 will be re-run in full.
- Remaining steps on resume, in order:
  1. `python -m scripts.run_experiment --config configs/1m/exp160_baselines.yaml`
  2. `python -m scripts.run_experiment --config configs/1m/exp300_mog.yaml`
- Pre-registered H2 decision (exp200): the free proxy has median ρ −0.33 and top-1 agreement 8.6% vs 12.1% chance, so **H2 is FALSIFIED**. The same-batch proxy has ρ 0.22. The online proxy selector is **NO-GO**. Full analysis follows after exp160 and exp300.

---

## Stage 3 results (2026-09-23). Full tables are in RESULTS.md.

### exp150 / exp151: LR tuning (seed 0)
- Result: every arm's best LR is inside its grid after the edge rule. Tuned LRs are in `configs/1m/exp160_baselines.yaml`.
- Reproducibility check: the atlas run (exp100) reproduced the tuning finals exactly (muon 1.6862, adamw 2.0280), and the exp160 re-run matched the aborted attempt (sgd seed 1: 2.4138). Training is deterministic, and the instrumentation does not perturb it.

### exp100: atlas (descriptive)
- (a) The r_eff trend runs against H2's sub-claim. Momentum r_eff/min(m,n) *rises* over training: Muon 0.16→0.36 with no block falling; AdamW 0.09→0.14 with 7% of blocks falling.
- (b) The free secant is 2.3–4.5× the same-batch secant, and block gradient SNR is below 1. The PDF's free curvature estimate is dominated by gradient noise at 2048 tokens/step.
- (c) The E_c argmax is "sign" in 77–96% of cells, but sign/elementwise arms lose by about 0.44 nats in exp160, so the proxy mis-ranks.

### exp200: oracle and proxy (H1, H2)
- H2 is **FALSIFIED** under the pre-registered rule. The free proxy has median ρ −0.33 and top-1 agreement of 8.6% against 12.1% chance (binomial p=0.87). The same-batch proxy has ρ 0.22, and neither passes.
- Gate: the online proxy selector is **NO-GO** and is not built.
- Exploratory, not pre-registered: the own-direction secant (one extra backward per candidate) reaches ρ 0.76 with top-1 of 48% (p=2e-15). Gradient-fit alone (ρ −0.87) and 1-step lookahead (ρ −0.86) anti-predict the 20-step oracle, which confirms the PDF's greedy-myopia risk.
- Oracle choices: 60 of 81 cells prefer a non-incumbent rule and 36 of those are stable on the alternate stream, but the gains are tiny (median stable gain 7e-5 nats). The consistent pattern is NorMuon on MLP up-projections (12/12 cells, 9 stable).

### exp160 + exp300: matched-token comparison (seeds 1–3)
- Spectral hidden-weight geometry beats AdamW by about 0.35 nats. NorMuon, Muon and Duality are tied.
- The oracle-scheduled MOG arm scores 1.7046 against NorMuon's 1.6897: Δ +0.0150 [−0.0010, +0.0310]. Against Muon it is +0.0119 [−0.0015, +0.0252]. **H1 is NOT SUPPORTED** at 0.82M: the pre-registered verdict is INCONCLUSIVE, and the point estimate is unfavourable.
- Interpretation: single-block, short-horizon oracle wins do not add up. Composing up to 12 simultaneous switches per phase gives a slightly worse run. Possible causes are block interactions (Prop. 3 coupling) and horizon myopia.
- Per-head Muon is worse than whole-matrix Muon (+0.032 [+0.012, +0.052]), which is partial evidence against H4. Adam-mini is unstable on seed 1 at its tuned LR.

### Gate decision: Stage 3 → 4
- **NO-GO for scaling MOG as specified in the PDF.** Its free proxy fails, and the oracle-derived assignment does not beat the tuned baselines.
- **REDESIGN before any scale-up.** Each item below needs a new pre-registration:
  1. H2′: the own-direction secant proxy (it cost one extra backward per candidate here), tested on fresh checkpoints and seeds.
  2. A longer-horizon oracle, plus a joint (multi-block) oracle, to test the additivity assumption.
  3. The random-partition control for H4.
  4. A realistic tokenizer for H5.
- The robust finding here is the well-known one: spectral geometry on hidden weights ≫ elementwise geometry at this scale.

---

## Redesign pre-registration (written 2026-09-24, before any redesign run)
All of these are additive: they use new experiment IDs and only *read* the earlier runs. Configs are in `configs/redesign/`, and the chain is `bash scripts/run_redesign.sh`. Setup matches Stage 3 unless stated otherwise.

### exp400 + exp410: H2′, the own-direction secant proxy (exploratory in exp200, now a test)
- A fresh incumbent (Muon hybrid at lr 0.01, **new seed 4**) writes checkpoints at 100/300/500. The exp200 oracle code runs on them with **fresh data streams** 787/788 and h=20.
- **SUPPORTED** if the own-direction proxy has median Spearman ρ ≥ 0.5 **and** a top-1 agreement above chance (one-sided binomial p < 0.05).
- **FALSIFIED** if median ρ ≤ 0.3 or top-1 agreement is not above chance. Anything in between is INCONCLUSIVE.
- Caveat stated up front: this proxy is not "free". It costs one extra backward per candidate per block.

### exp420: additivity and horizon of oracle switches
- Uses the exp200 stable switches and the exp100 checkpoints, read-only, on a fresh stream 779 at horizons h ∈ {20, 100}.
- Additivity holds if the joint gain is > 0 and ≥ 0.5 × the sum of the individual gains.
- Horizon persistence is the fraction of individual gains still positive at h=100.
- Reported per checkpoint. With only 3 checkpoints, no significance claim is made.

### exp430 + exp431: H4 random-partition control
- `randpart` uses head-sized groups (k = d/H) on attention with a fixed random row/column membership per block. Its LR is tuned on seed 0 with the same 3-point rule, then it is evaluated on seeds 1–3.
- Compared pairwise with the exp160 `headmuon` and `muon` runs (same seeds, same code paths).
- The symmetry-matching part of H4 is **SUPPORTED** if headmuon < randpart with a 95% CI excluding 0. It is **FALSIFIED** if the CI includes 0 or randpart is better.
- The other clause of H4 ("beats whole-matrix Muon") already failed in exp160.

### exp440: H5 on a real rare-token regime
- GPT-2 BPE remapped to the corpus (~11.7k types, 3.6k singletons), about 3.8M params, 300 steps (~2 epochs), 16 eval batches.
- Arms: `muon` (Adam on embed/pos/head) vs `duality` (per-token rows on embed/pos/head). The hidden weights are identical.
- Each arm's LR is tuned on seed 0 over {0.0033, 0.01, 0.03}, then evaluated on seeds 1–3.
- **SUPPORTED** if duality < muon with a paired 95% CI excluding 0 **and** the mean gain in the rarest train-frequency decile of target tokens exceeds the gain in the most frequent decile.
- **FALSIFIED** if there is no gain (CI includes 0 or duality is worse), or if the gain is not larger on rare tokens.

---

## Redesign results (2026-09-24). Full numbers are in RESULTS.md, "Redesign experiments".
All runs have clean git state: exp400 (432 s), exp410 (4335 s), exp420 (1262 s), exp430/431 (396 s each), exp440 (2114 s). The CPU ran about 2.5× slower than on 2026-09-23, probably from throttling. That changes wall time only, not results, because training is deterministic.

- **H2′ SUPPORTED.** The own-direction proxy reaches median ρ 0.771 and top-1 42.0% vs 12.1% chance (p=1.6e-11) on a fresh seed and fresh streams. The free-proxy failure (ρ −0.267) and the negative ρ of gradient-fit and lookahead (−0.88) both replicate.
- **Additivity/horizon.** At h=20 the joint gain ≈ the sum of single gains (steps 100 and 500). At step 300 the exp200 "stable" switches are net negative on stream 779, both singly and jointly. At h=100 the share of single-block gains that stay positive drops at every checkpoint. Interpretation: H1 failed because per-block effects are near the noise floor and short-lived, *not* mainly because of interactions.
- **H4 FALSIFIED.** Per-head vs random head-sized groups: −0.024 [−0.060, +0.012], so the CI includes 0. Every grouped variant is worse than whole-matrix Muon.
- **H5 FALSIFIED.** On BPE, per-token rows lose to Adam on vocab blocks (+0.052 [+0.016, +0.088]). The loss is higher in every frequency decile, with no rare-token advantage.
- **Gate.** The only surviving positive for MOG's selection idea is H2′. The next pre-registration, if any, is an H2′-scored selector, charged for its extra backward passes and compared with NorMuon at matched wall-clock. It needs the user's go-ahead.
