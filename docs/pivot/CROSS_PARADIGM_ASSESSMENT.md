# Cross-paradigm geometry: assessment and plan

Source document: *Does optimizer geometry follow a predictable rule across learning paradigms?* (literature study and proposal, 18 pp., 23 Sep 2026, no experiments). This is a review of it against MOG's measured results, plus a staged plan. Nothing here is a new result.

## 1. What the document proposes

**Core claim (its H1, "sufficiency").** At matched per-block update scale, the benefit of a block geometry depends on the learning paradigm (supervised, SSL, diffusion, RL) *only* through a few measurable block statistics:
- signal rank and SNR (split-batch alignment);
- curvature–Gram coupling exponent p;
- curvature axis-alignment κ∞;
- drift rate and noise tail index;
- symmetry class, which only restricts the admissible norms.

**Test.** Leave-one-paradigm-out prediction of per-block paired-oracle gain. It passes if held-out Spearman ρ ≥ 0.5 and paradigm labels add nothing. It fails if ρ ≤ 0.3.

**Gate the document puts first (its H2, "shape beyond scale").** A geometry effect must survive per-block RMS matching and beat a random-spectrum (Kaon) control. Otherwise "geometry" effects are step-size effects, and the project pivots to a per-block scale theory.

**Theory contributions it labels potentially novel:**
- robust-Fisher unification of the norm lattice (Thm R1);
- the coupling law α* = 1 − 4p (Thm C1);
- a spectral-vs-elementwise inequality through κ∞;
- low-SNR constants and a low-rank alignment ceiling;
- the tail-amplification trade-off;
- mappings PPO ↔ spectral and TRPO ↔ Fisher;
- the SSL collapse-mechanism table;
- diffusion damped whitening and weighting invariance.

**Citations checked here (arXiv abstracts, 2026-09-24).**
- Kaon (2605.11181) exists: random singular values match Muon on GPT-2.
- Ruan et al. (2607.16169) exists: an RMS-matched control removes Muon's agentic-RL gain.
- Lau & Su (2605.18106) exists: the symmetry-compatible stack is published.

The document's framing of these three papers is accurate.

## 2. What MOG's own data already says about it

| Document claim | MOG evidence (0.82M char/BPE LM, CPU) | Bearing |
|---|---|---|
| Geometry effects may be pure scale (R1/H2) | Every LMO rule is rescaled to per-block RMS 0.2 (`mog/optim/blockwise.py`). At that matched RMS with a tuned LR, spectral hidden weights (1.693) beat sign (2.128), rows (2.127) and normalized-SGD (2.521) by 0.43 nats or more. | **Partial evidence that shape matters** in supervised LM. Missing: random-spectrum arm and RMS-matched Adam. Both are cheap to add. |
| The "free" selection signal is noise-biased; replace it with split-batch alignment | exp100: free secant is 2.3–4.5× the same-batch secant. exp200/410: free proxy ρ −0.27 to −0.33. | **Agrees.** |
| Alignment statistics predict benefit | exp200/410: gradient-fit θ/C alone gives ρ −0.87/−0.88. The own-direction *curvature* gives ρ 0.77. | **Warning.** In MOG the alignment term was anti-predictive and curvature along the actual step carried the signal. Its §4.8(1) concedes this. H1's statistic list is mostly alignment-type. |
| E3: whole-matrix Muon on multi-head W_Q is "excess invariance", so per-head should help | exp160/431: per-head is worse than whole-matrix (+0.032) and no better than random groups. | **Contradicted at small scale.** |
| R1: the worst-case Fisher norm for one-hot inputs is per-token (embeddings) | exp440 (BPE): per-token rows lose to Adam, +0.052 [+0.016, +0.088], with no rare-token advantage. | **Contradicted.** The worst-case norm is not the best-performing rule. |
| Muon keeps rank high by injecting tail directions | exp100: momentum r_eff rises 0.16→0.36 under Muon vs 0.09→0.14 under AdamW. | **Consistent** (descriptive, 1 seed). |
| The per-block oracle gain is the prediction target | exp200/420: median per-block gain is 7e-5 nats over 20 steps. It fades by 100 steps and partly fails to replicate. | **Biggest risk.** If the target sits at the noise floor, ρ ≥ 0.5 across paradigms may be unreachable for statistical reasons, not scientific ones. |

## 3. Assessment

**Strengths**
- It demotes symmetry to a constraint and puts the scale confound first.
- It pre-registers falsifiers, with an outcome map where every branch is reportable.
- It reuses exactly the MOG tooling that survived: atlas, paired oracle, pre-registered proxy validation.
- Its statistics section is sound: 5 seeds detect about 2 SD, TOST for nulls, IQM for RL.

**Weaknesses and risks**
1. **The target may be too small to predict.** See section 2. Other paradigms, particularly RL with low SNR, may have larger per-block effects, or may not. The first measurement must be the *size* of per-block gains per paradigm, before any predictor is fitted.
2. **The statistic set is untested and partly contradicted.** MOG found alignment anti-predictive. The own-direction secant (H2′) should be a reference predictor alongside the proposed statistics, even though it is not "free".
3. **Theory is unreviewed.** R1 and C1 are sketches under idealizations. R1's prescriptions already failed twice in MOG (embeddings, per-head).
4. **Compute does not fit the plan as written.** The 16-week plan assumes GPU. This machine has 4 CPU threads and no GPU. CIFAR-10 flow matching to meaningful FID, RLVR, and 10-seed width-≥256 MinAtar PPO are not feasible here. Small proxies are needed, and the plan must say so.
5. **Novelty is concentrated in one claim:** the leave-one-paradigm-out *predictive* test. Most of the theory table is "unclear / potentially novel" and would need a literature re-check before any write-up.

**Verdict.** The pivot is sound in direction and better framed than the original PDF. It should start with the cheap gate the document itself names (H2: shape vs scale), run on the existing 0.82M LM, because MOG already has 90% of the code. If shape does not survive, the project becomes the per-block scale theory, and the costly cross-paradigm work is skipped.

## 4. Plan (each phase gated; nothing runs without approval)

**Phase 0: shape-vs-scale gate on the existing LM (CPU, about 1–2 h).** Tests the document's H2.
- Add two rules to `BlockOptimizer`:
  - `randspec`: U·diag(random)·Vᵀ, the Kaon-style control, rescaled to RMS 0.2;
  - `adam_rms`: Adam direction rescaled to per-block RMS 0.2.
- Run arms muon hybrid, randspec hybrid and adam_rms hybrid against the existing AdamW and Muon results. Use the same 3-LR grid, tune on seed 0, evaluate on seeds 1–3, and add seeds 4–5 for power.
- Also run a *per-block* paired-oracle version (spectral vs randspec vs adam_rms at common checkpoints, reusing `experiments/03_oracle`). This is the lower-variance test the document recommends.
- GO if spectral beats randspec and adam_rms with a CI excluding 0 (end-to-end), or in paired oracle contrasts for at least one block class. NO-GO if TOST shows equivalence: the project then pivots to per-block scale.

**Phase 1: estimators on synthetic blocks (CPU, days).** Implement and validate against the document's closed forms (its Figs. 3–4):
- split-batch alignment;
- r_eff of split-batch momentum;
- the coupling exponent p;
- κ∞ via SDP.

Add them to the atlas.

**Phase 2: per-paradigm effect-size survey, CPU-scale proxies.** Measure the *size* of per-block paired-oracle gains before fitting anything. Four settings:
- supervised: small CNN on CIFAR-10 subset;
- SSL: tiny SimCLR on the same data;
- generative: tiny flow-matching MLP/U-Net on MNIST or 8×8 data;
- RL: PPO on MinAtar or CartPole-class tasks, width 256.

Gate: if gains are at the noise floor in every paradigm, stop H1 and report.

**Phase 3: sufficiency test (H1).** Collect per-block (statistics, gain) pairs across the four paradigms. Fit leave-one-paradigm-out with the pre-registered thresholds. Reference predictors are the H2′ own-direction secant (upper bound) and condition number (negative baseline).

**Phase 4: mechanism tests,** chosen by what Phase 2 shows: H3 tail-amplification, H5 actor–critic rank, H6 SSL 2×2, H7 per-σ diffusion. GPU-scale items (RLVR, CIFAR FID, DiT) stay out of scope unless compute is added.

## 5. Decisions for the owner

1. Approve Phase 0? It is the only thing proposed to run now.
2. Where should the cross-paradigm work live? Options are a new package area in this repo on the pivot branch, or a separate repository.
3. Is GPU compute available later? That decides whether Phase 4 includes RLVR, DiT or CIFAR FID.
