# Research problem statement: optimizer geometry beyond supervised transformer LMs

*This is a separate project spun out of MOG. MOG itself stays on supervised, autoregressive transformer language models.*

## 1. Motivation

MOG treats an optimizer as a choice of norm per parameter block: elementwise (Adam/sign), row/column, grouped spectral, or whole-matrix spectral (Muon). Its measurements are made on supervised next-token prediction. That setting has two properties the rest of the geometry argument silently relies on: a **stationary objective**, and **dense 2-D projections acting on an RMS-normalized residual stream**.

Other learning paradigms break one or both properties. That makes the question "which geometry does each block want?" a different question there, not just the same question on a new dataset.

## 2. Why each domain is its own problem

| Domain | What changes for geometry | Open question |
|---|---|---|
| **Reinforcement learning** (policy gradient, actor-critic, value learning) | The objective is non-stationary: the target moves with the policy and the data distribution depends on the parameters. Gradient noise is very high and heavy-tailed. The actor and critic are separate networks with different curvature. Trust regions (PPO/TRPO) already constrain the step in a *distribution-space* norm (KL), not a weight-space norm. | Is the right weight-space geometry the one that best approximates the KL trust region (the Fisher geometry) per block? Does spectral normalization help or hurt plasticity and loss-of-plasticity effects? Should actor and critic use different geometries? *We found no published Muon-for-RL study in a 2026-09 search, so this is a genuine gap.* |
| **Unsupervised / self-supervised** (contrastive, MAE, JEPA, VAEs) | Losses have collapse modes: representations that shrink to low rank. That is exactly what spectral geometry acts on, because it flattens the update spectrum. Projector and predictor heads are small and critical. | Does a spectral geometry *prevent* dimensional collapse (a mechanism claim), and does per-block geometry selection detect the onset of collapse via r_eff? |
| **Generative diffusion / flow matching** | The loss is a weighted sum over noise levels, so the gradient statistics mix very different regimes. U-Nets mix convolutions and attention. | Is the optimal geometry noise-level dependent? A single step cannot switch per sample, but does an average per-block geometry exist? A controlled benchmark exists (arXiv:2609.23055, verified 2026-09-23). |
| **Convolutional networks** | A conv kernel is a 4-D tensor. Which 2-D reshape (out × in·k·k, or per-spatial-offset blocks) is the "operator" is itself a geometry choice. Muon-C (arXiv:2609.09676, verified 2026-09-23) proposes an operator-aligned variant. | This is the grouping axis 𝒢 of MOG applied to kernels. Is the best reshape predictable from the symmetry principle (§3.2 of the PDF)? |
| **Graph neural networks, state-space models, MoE routers** | Weight sharing across nodes and time, recurrent state matrices, and discrete routing. | The symmetry principle predicts specific groupings, such as per-expert blocks and per-state-channel rows. Are they right? |

## 3. Candidate central question for the separate project

> Across learning paradigms (supervised, self-supervised, generative, reinforcement), does the per-block optimal update geometry follow a *predictable* rule? Candidates are the architecture's symmetry group, the gradient spectrum, or the Fisher/KL geometry. Or is the geometry advantage of spectral methods specific to supervised dense-projection training?

Falsifiable sub-hypotheses:
1. **(RL)** Per-block geometry matched to the Fisher approximation beats both Adam and Muon on sample efficiency at a matched update budget. Falsified if neither geometry-aware arm beats tuned Adam across ≥5 seeds on ≥3 environments.
2. **(SSL)** Spectral geometry on encoder blocks reduces dimensional collapse, measured by the representation's effective rank. Falsified if representation r_eff is the same under Adam and Muon at matched downstream accuracy.
3. **(Conv)** The symmetry-principle grouping for conv kernels matches the oracle-best reshape. Falsified if the oracle's choice is no better than a random reshape.

## 4. What transfers from MOG unchanged

- The geometry interface (`lmo`, `norm`, `dual_norm`, `C`) and the per-block optimizer.
- The **atlas** (measurement), the **oracle** (short-rollout ground truth) and **proxy validation**. This measurement methodology is domain-agnostic, and it is the main thing MOG contributes to a cross-domain study.

## 5. What does not transfer

- The data pipeline, the evaluation metrics (returns and sample efficiency, not val loss) and the seed counts. RL needs ≥10 seeds.
- The oracle's short-horizon rollout. In RL, a 20-step rollout changes the data distribution, so the "freeze everything else" control is weaker.

## 6. Minimal first experiments (CPU-feasible)

1. A small-CNN block atlas on CIFAR-10 (conv kernels with 3 reshapes).
2. PPO on 3 classic-control or MinAtar environments with {Adam, Muon-hidden, row-norm}, 10 seeds each.
3. A SimCLR-style small encoder: representation r_eff under Adam vs Muon.
