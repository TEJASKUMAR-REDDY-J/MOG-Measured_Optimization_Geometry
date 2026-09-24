# RESULTS

Only measured results appear here. Each links to its run directory under `results/raw/`. Status keys: **PASS**, **FAIL**, **INCONCLUSIVE**, **NOT TESTED**.

Setup for every Stage 3 run:
- **Model and data:** 0.82M-parameter GPT (d=128, 4 layers, 4 heads, ctx 128) trained on char-level Tiny Shakespeare. That is 600 steps × 2048 tokens (1.23M tokens), with val loss measured on 8 fixed held-out batches.
- **Hardware:** CPU only.
- **Tuning budget:** one tuned scalar (the LR) per arm, on a 3-point grid plus the pre-registered edge rule. Tuning used seed 0 only; everything reported below uses seeds 1–3.
- **Intervals:** paired 95% t-interval (df=2).

## Synthetic and theory checks (evidence class `synthetic`)

| Check | Status | Evidence |
|---|---|---|
| PDF §4.3 closed-form columns (r_eff; full-spectral θ/C) | PASS | Unit test. Exact values: 512, 281.65, 28.28, 5.30 and 1.000, 0.5501, 0.0552, 0.0104 |
| PDF §4.3 Monte Carlo columns (rows, elementwise), 5 seeds | PASS (all within 2 seed-std) | `exp000_synthetic_gradient_fit/20260923T104858277028Z` |
| Fig 1c: θ_k/C_k falls monotonically in k, and is flat at β=0 | PASS | same run |
| NS nuclear-norm proxy (Prop. 2 assumes it is exact) | **FAIL** (biased) | same run. Error ranges from −2% to −22%. The direction cosine to the exact $UV^\top$ falls to 0.63 at β=1.5. |

## Stage 3: baselines at matched tokens (exp160, exp300)

Runs: `exp160_1m_baselines/20260923T161011974994Z` and `exp300_mog_1m/20260923T180330693402Z`.

| Arm | Hidden-weight rule | LR | Val loss, mean ± sd (seeds 1/2/3) | Δ vs Muon hybrid [95% CI] |
|---|---|---|---|---|
| NorMuon hybrid | normuon | 0.01 | **1.6897** ± 0.0049 | −0.0031 [−0.0236, +0.0174] |
| Muon hybrid (folklore) | spectral | 0.01 | 1.6928 ± 0.0035 | — |
| Duality (PDF §2.2: per-token rows on vocab) | spectral | 0.01 | 1.6938 ± 0.0078 | +0.0011 [−0.0266, +0.0287] |
| Scion-style (sign head) | spectral | 0.01 | 1.7000 ± 0.0050 | +0.0072 [+0.0031, +0.0113] |
| **MOG, oracle-scheduled** | spectral + oracle switches | 0.01 | 1.7046 ± 0.0049 | +0.0119 [−0.0015, +0.0252] |
| Muon on every matrix | spectral | 0.01 | 1.7159 ± 0.0042 | +0.0232 [+0.0190, +0.0274] |
| Per-head Muon | spectral_head (attn) | 0.01 | 1.7247 ± 0.0049 | +0.0319 [+0.0121, +0.0517] |
| Partial whitening α=½ | pw05 | 0.0033 | 1.9270 ± 0.0102 | +0.234 |
| AdamW | adam | 0.01 | 2.0429 ± 0.0231 | +0.350 [+0.289, +0.411] |
| Row-norm everywhere | rows | 0.003 | 2.1268 ± 0.0106 | +0.434 |
| Signum | sign | 0.003 | 2.1283 ± 0.0141 | +0.436 |
| Lion | lion | 0.001 | 2.1938 ± 0.0150 | +0.501 |
| Adam-mini | adam_mini | 0.01 | 2.2518 ± 0.2300 (seed 1: 2.514, unstable) | +0.559 [−0.021, +1.139] |
| SGD (momentum) | sgd | 0.3 | 2.4100 ± 0.0055 | +0.717 |
| Normalized SGD | euclid | 0.003 | 2.5211 ± 0.0669 | +0.828 |

What the table shows:
- **The geometry family used on hidden weights dominates everything else.** Spectral (Muon-family) hidden weights beat AdamW by about 0.35 nats, which is far outside the noise.
- **Within the spectral family the choice barely matters.** NorMuon, Muon and Duality are statistically tied.
- **These variants are worse than the plain Muon hybrid:**
  - Per-head grouping: +0.032
  - Spectral on the embedding and head: +0.023
  - A sign-update head: +0.007
  - α=½ partial whitening: +0.23
- **Adam-mini at its tuned LR is unstable on one of three seeds.**

## Stage 3: atlas (exp100, 1 seed, descriptive)

Run: `exp100_atlas_1m/20260923T133718268064Z`.

| Readout (pre-declared) | Muon incumbent | AdamW incumbent |
|---|---|---|
| Mean momentum r_eff/min(m,n), first → last measurement | 0.16 → 0.36; 0% of blocks fall | 0.09 → 0.14; 7% of blocks fall |
| Median free secant ÷ same-batch secant | 3.5–4.5× | 2.3–2.5× |
| Proxy argmax (same-batch E_c) | sign in 286/297 cells | sign in 230/297 cells |
| NS dual-norm error / direction cosine (median) | −12% / 0.97 | −10% / 0.83 |
| Block gradient SNR (median) | 0.33 | 0.65 |

## Stage 3: oracle and proxy (exp200)

Run: `exp200_oracle_1m/20260923T134338543058Z`. The oracle covers 3 checkpoints × 27 blocks = 81 (checkpoint, block) cells.

- The oracle preferred a non-incumbent rule in 60 of 81 cells. In 36 of those, the gain was still positive on a second data stream.
- Gains are small: the median stable gain is 7e-5 nats per block over 20 steps, and the maximum is 5.6e-3 (the LM head).
- The most consistent choice is **NorMuon on the MLP up-projections**: best in 12 of 12 cells, and stable in 9.

| Proxy | Median Spearman ρ vs oracle | Top-1 agreement (chance 12.1%) | One-sided binomial p |
|---|---|---|---|
| E_c free (PDF §5) | **−0.33** | 8.6% | 0.87 |
| E_c same-batch | 0.22 | 6.2% | 0.97 |
| E_c own-direction (one extra backward per candidate) | **0.76** | 48.1% | 2e-15 |
| θ/C gradient-fit only | −0.87 | 1.2% | — |
| 1-step lookahead | −0.86 | 3.7% | — |

## Redesign experiments (pre-registered 2026-09-24)

| ID | Test | Result | Verdict |
|---|---|---|---|
| exp400 + exp410 | **H2′**: own-direction secant proxy on fresh checkpoints (seed 4) and fresh streams 787/788 | Median ρ **0.771** (IQR 0.62–0.87). Top-1 agreement 42.0% vs 12.1% chance (binomial p=1.6e-11). Mean regret 3.1e-4. For comparison: free proxy ρ −0.267 (top-1 16%, p=0.18); same-batch ρ 0.19; gradient-fit alone ρ −0.88; 1-step lookahead ρ −0.88. | **SUPPORTED** (replicates exp200). The H2 falsification of the free proxy also replicates. |
| exp420 | Additivity and horizon of the exp200 stable switches (stream 779) | h=20 joint vs sum of single-block gains: step 100 +0.0046 vs +0.0028; step 300 −0.0007 vs −0.0007; step 500 +0.0007 vs +0.0007. h=100: +0.0091 vs −0.0055; −0.0056 vs −0.0045; +0.0003 vs +0.0001. Share of single gains still positive at h=20 → h=100: 58%→25%, 67%→42%, 100%→75%. | Additivity **holds** at 2 of 3 checkpoints (step 300 fails at both horizons). Gains **fade** with horizon, and some "stable" exp200 switches do not replicate on a fresh stream. |
| exp430 + exp431 | **H4 control**: head-sized groups with random membership (randpart), seeds 1–3 | randpart 1.7488 (1.7686 / 1.7358 / 1.7419). headmuon − randpart: −0.0241 [−0.0598, +0.0115]. randpart − muon: +0.0560 [+0.0040, +0.1081]. | Symmetry-matching part **FALSIFIED** (the CI includes 0). Grouping of any kind is worse than whole-matrix spectral. |
| exp440 | **H5**: Adam vs per-token rows on embed/pos/head with GPT-2 BPE (11,706 types), seeds 1–3 | muon 4.8243 vs duality 4.8762. duality − muon: +0.0519 [+0.0160, +0.0877]. Per train-frequency decile of the target token (duality − muon): rarest +0.094 [−0.134, +0.322], median +0.073 [+0.039, +0.107], most frequent +0.059 [−0.028, +0.147]. | **FALSIFIED**. Per-token rows are *worse* than Adam, and the deficit is not smaller on rare tokens. |

## Hypotheses

| # | Status | Evidence |
|---|---|---|
| H1 | **NOT SUPPORTED at 0.82M** (INCONCLUSIVE by the pre-registered criterion). The point estimate is unfavourable. | The oracle-scheduled MOG arm is +0.0150 [−0.0010, +0.0310] vs the best baseline (NorMuon) and +0.0119 [−0.0015, +0.0252] vs Muon. exp420 shows why: per-block gains sit at the noise floor, fade by h=100 and partly fail to replicate, although they combine roughly additively at h=20. |
| H2 | **FAIL** (pre-registered: the free proxy's median ρ ≤ 0.3 and its top-1 agreement is not above chance) | exp200, exp410 (replicated) |
| H2′ (own-direction proxy) | **PASS** (ρ 0.771, top-1 42%, p=1.6e-11). It is not free: it costs one extra backward per candidate. | exp410 |
| H2 sub-claim: r_eff collapses over training | **FAIL at this scale** | exp100: r_eff *rises* under both incumbents |
| H3 | NOT TESTED | — |
| H4 | **FAIL** | Per-head is worse than whole-matrix Muon (+0.032 [+0.012, +0.052]) and not distinguishable from random groups (−0.024 [−0.060, +0.012]) |
| H5 | **FAIL** | With BPE (exp440), per-token rows are worse than Adam on vocab blocks: +0.052 [+0.016, +0.088], with no rare-token advantage |
| H6–H9 | NOT TESTED | — |
