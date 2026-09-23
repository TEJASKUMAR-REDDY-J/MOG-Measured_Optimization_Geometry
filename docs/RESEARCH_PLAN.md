# RESEARCH PLAN

## 1. Central question

Should the update geometry of each neural-network parameter block be **selected from measurement** instead of fixed by convention? Today's convention is Muon on 2-D hidden weights and AdamW on everything else.

The PDF's problem statement asks three questions:
1. Is the optimal update geometry heterogeneous across the blocks of a transformer?
2. Can the right geometry per block be identified online from quantities already available during training?
3. Does acting on that identification improve the loss/compute frontier over the best homogeneous optimizer at matched budget?

## 2. Subquestions, in dependency order

| # | Subquestion | Answered by | Blocks |
|---|---|---|---|
| Q0 | Are the PDF's closed-form and synthetic computations correct? | Exp 0 plus unit tests | All |
| Q1 | Is our implementation of each geometry correct as an optimizer? | Unit tests, Muon equivalence, Stage-2 sanity training | All training |
| Q2 | Do real momenta show *measurable* per-block heterogeneity in $r_{\rm eff}$, $\theta_c/C_c$, the secant $\hat L_c$ and $E_c$? | Exp 1 (atlas) | Q3–Q6 |
| Q3 | Is the *true* short-horizon best geometry heterogeneous across blocks (oracle)? | Exp 2 | Q4, Q5 |
| Q4 | Does the free proxy $E_c$ predict the oracle? | Exp 2 (proxy vs oracle) | Q5 |
| Q5 | Does online selection with hysteresis and damping beat tuned homogeneous baselines at matched budget? | Exp 3 | Q6 |
| Q6 | How do the effects scale with model size, batch size, width and depth? | Exp 5 | — |

The theory-level null result is valid: heterogeneity may be real while the folklore two-bucket assignment already captures almost all of it. Every experiment is designed so that this outcome is reportable.

## 3. Hypotheses

H1–H9 are in `HYPOTHESES.md`. They map onto the subquestions as follows: H1 → Q3 (and Q5), H2 → Q4 (and Q2's $r_{\rm eff}$ trend), H3/H6 → Q6, H4/H5/H8/H9 → targeted interventions after Q3, H7 → Q5 (damping).

## 4. Experiment ladder and decision gates

A later stage starts only when the previous gate is **GO**. For every gate we write a GO / NO-GO / REDESIGN entry in `EXPERIMENT_LOG.md`. Scaling up requires all of the following:
- the implementation is numerically correct
- the signal is detectable
- the result reproduces
- the result survives ≥3 seeds
- the larger scale answers a question the smaller one cannot

| Stage | Scale | Purpose | Experiments | GO criteria | NO-GO / REDESIGN triggers |
|---|---|---|---|---|---|
| 0 | none | Mathematical and unit validation | Unit tests; Exp 0 (synthetic) | All tests pass. Exp 0's closed-form values match the PDF, and the Monte Carlo columns match to within stated seed spread, or any disagreement is explained. NS proxy error is quantified. | An identity fails, or a PDF table value that cannot be reproduced and cannot be explained → record it and correct THEORY.md before any training. |
| 1 | synthetic | Synthetic matrix experiments beyond the PDF: rank, condition number, size, non-square shapes, grouping | Exp 0 extensions | The qualitative trends of §4.3 hold across sizes and shapes. | Trends reverse at non-512 shapes → the proxy is scale-fragile; redesign before Stage 2. |
| 2 | ~0.1–1M params (tiny MLP / transformer) | Pipeline verification only. **No scientific claims.** | Optimizer sanity (all geometries as standalone optimizers); atlas instrumentation dry run | Every arm trains stably at some LR. Metrics are finite and logged. Runs are bit-reproducible at a fixed seed. Instrumentation overhead is measured. | Instability that cannot be traced to a hyperparameter → debug. |
| 3 | ~1M transformer | First meaningful measurements | Exp 1 (atlas), Exp 2 (oracle + proxy), baselines (AdamW, Muon, hybrid) | Atlas shows between-block dispersion in $E_c$ **larger than seed-to-seed dispersion**. The oracle argmax differs from the folklore assignment in some blocks by more than rollout noise, or the null is recorded. Proxy-vs-oracle is measured with CIs. | Heterogeneity is within noise → record H1 as INCONCLUSIVE or FALSIFIED at this scale. Scale up only if a specific reason to expect scale-dependence exists and is written down. |
| 4 | ~5M | Repeat only informative experiments. Exp 3 (online) only if the proxy passed. | Exp 1/2 subset; Exp 3 if justified; first ablations | The effect survives at 5M with ≥3 seeds. | The effect vanishes → stop scaling that branch. |
| 5 | ~20–50M | Scaling, overhead, stability | Exp 3–5 subset | The gain or finding persists and the overhead is acceptable. | — |
| 6 | ~100–124M | Paper-scale design (PDF §7), only with evidence | Full atlas, oracle, proxy, online, Pareto, multi-seed | — | — |

Compute constraint (current machine): **CPU only, no CUDA device detected** (torch 2.8.0+cpu). Stages 0–3 fit on CPU. Stage 4 is marginal on CPU. Stage 5 and above need a GPU, and that has to be arranged before those stages. It is not a reason to skip gates.

## 5. First three experiments (proposed; NOT run)

1. **exp000_synthetic_gradient_fit** (Stage 0). Reproduce the PDF §4.3 table and Figs. 1b/1c/1d/2a. Measure the bias of the NS-based nuclear norm (float32 and bf16). Config: `configs/synthetic/exp000_gradient_fit.yaml`. Runs in seconds on CPU.
2. **exp010_tiny_optimizer_sanity** (Stage 2). A tiny transformer (~0.3M params) on a small character-level corpus. Arms: AdamW, Muon (hidden) + AdamW (rest), and each geometry as a standalone optimizer. Each arm gets a small LR sweep with an identical sweep budget, 3 seeds. Purpose: implementation correctness and stability. **No hypothesis is tested.** Needs a dataset download, which will be requested before running.
3. **exp011_tiny_atlas_dryrun** (Stage 2). Instrument the same tiny model. Every N steps, log $r_{\rm eff}$, $d_{\rm eff}$, $m_{\rm eff}$, all $\theta_c/C_c$, secant $\hat L_c$ (same-batch vs consecutive-batch variants, to test THEORY §5.5), NS proxy error on real momenta, per-group SNR and instrumentation overhead. Purpose: validate the atlas pipeline and check whether the §4.3 synthetic regime (β, $r_{\rm eff}$) matches real momenta. **Not** evidence for H1.

## 6. Anti-bias commitments

- Pre-register: write the log entry (hypothesis, metric, falsification criterion, analysis plan) *before* running each experiment.
- Analyze all blocks, all seeds and all pre-declared checkpoints. No post-hoc subset selection.
- Report every tuned scalar per arm, and give every arm an equal tuning budget.
- Keep `RESULTS.md` to measured results only. Synthetic and closed-form checks are labeled as such.
- Record negative results with the same prominence as positive ones.
