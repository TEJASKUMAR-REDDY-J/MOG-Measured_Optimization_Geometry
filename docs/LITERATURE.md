# LITERATURE

Verification method: on 2026-09-23 we queried the arXiv export API
(`export.arxiv.org/api/query?id_list=...`) for every identifier and compared
the returned titles and authors with the PDF. For sources whose specific
*content* the PDF relies on, we also read the abstract or HTML page.

"Verified" means the identifier resolves to the stated paper. It does **not**
mean we independently reproduced the paper's results.

"Content check" records how the specific claim was checked. "Abstract" means
we read the arXiv abstract. "HTML (summarized)" means the full HTML text was
retrieved through a summarizing fetch tool. Claims marked that way should get
a manual re-read before they are cited in any write-up.

## A. Sources the PDF lists as "retrieved and verified"

| Title | Authors | Year | Identifier | Relevance | Exact claim used by PDF / MOG | ID verified | Content check |
|---|---|---|---|---|---|---|---|
| Gluon: Making Muon & Scion Great Again! (Bridging Theory and Practice of LMO-based Optimizers for LLMs) | Riabinin, Shulgin, Gruntkowska, Richtárik | 2025 | arXiv:2505.13416 | Closest theoretical home: layer-wise LMO steps | Analyses layer-wise LMO under *layer-wise (L0,L1)-smoothness*. Plots the secant "trajectory smoothness" $\hat L=\|\nabla f_{k+1}-\nabla f_k\|_\star/\|X_{k+1}-X_k\|$. | yes | HTML (summarized): the smoothness name and the secant formula were confirmed. **The PDF's "ICML 2026" venue was NOT found** on arXiv. |
| Why Muon Outperforms Adam: A Curvature Perspective | Wang, Zhang, Li, Bergemann, et al. | 2026 | arXiv:2606.04662 | H8 motivation | PDF: the gap "mainly comes from boundary layers". | yes | The abstract emphasizes *within-layer* curvature (NDS). HTML (summarized): Appendix C says the within-layer sharpness gap is concentrated in the first and deepest layers. **Partially supported and overstated** (THEORY.md D4). |
| NorMuon: Making Muon more efficient and scalable | Li, Liu, Liang, Chen, et al. | 2025 | arXiv:2510.05491 | Per-neuron scale on top of Muon; a baseline | Neuron-wise second-moment statistics and row-wise normalization after orthogonalization. Memory comparable to Muon. | yes | Abstract: confirmed. The abstract reports gains of 21.74% over Adam and 11.31% over Muon at 1.1B; these were not reproduced. |
| Drop-Muon: Update Less, Converge Faster | Gruntkowska, Maziane, Qu, Richtárik | 2025 | arXiv:2510.02239 | Granularity as a systems variable | Updates a random subset of layers each step and reaches the same accuracy up to 1.4× faster in wall-clock time. | yes | Abstract: confirmed. |
| Spectral Lens: Activation and Gradient Spectra as Diagnostics of LLM Optimization | Liu, Paquette, Sous | 2026 | arXiv:2605.05683 | Measurement substrate for Exp 1 | Activation covariance and per-sample gradient SVD spectra used as diagnostics. | yes | Abstract: confirmed. |
| Practical Efficiency of Muon for Pretraining | Essential AI (Shah, Polloreno, Stratos, et al.) | 2025 | arXiv:2505.02222 | Muon at scale, batch-size tolerance | Background only. | yes | Title only. |
| Muon Learns More Robust and Transferable Features than Adam | Ruan, Zhang, Wang, Zhang | 2026 | arXiv:2606.09658 | Background | Background only. | yes | Title only. |
| Convergence Analysis of Muon-type Methods with Inexact LMO in the Degenerate Case | Qian, Richtárik | 2026 | arXiv:2606.21581 | Inexact NS LMO (relevant to THEORY D8) | Background only. | yes | Title only. |
| MuonQ: Enhancing Low-Bit Muon Quantization via Directional Fidelity Optimization | Su, Zhang, Liu, Zhao, et al. | 2026 | arXiv:2605.11396 | Numerics risk; directional fidelity | Background only. | yes | Title only. |
| MuonRec: Shifting the Optimizer Paradigm Beyond Adam in Scalable Generative Recommendation | Shan, Yu, Chen, Cai, et al. | 2026 | arXiv:2603.00416 | Background | Background only. | yes | Title only. |
| Muon: An optimizer for hidden layers in neural networks (blog) | K. Jordan | 2024 | kellerjordan.github.io/posts/muon/ ; code github.com/KellerJordan/Muon (MIT) | Reference Muon implementation | NS quintic coefficients (3.4445, −4.7750, 2.0315), Nesterov `lerp` momentum, and shape factor max(1, m/n)^0.5. | yes (link from the repo README) | Code fetched from `muon.py`. Our optimizer matches it bit-for-bit (`tests/test_muon_equivalence.py`). |

## B. Sources the PDF cites "from background knowledge". Identifiers resolved by MOG.

| Title | Authors | Year | Identifier | Used for | ID verified |
|---|---|---|---|---|---|
| Decoupled Weight Decay Regularization | Loshchilov, Hutter | 2017 | arXiv:1711.05101 | AdamW baseline | yes |
| signSGD: Compressed Optimisation for Non-Convex Problems | Bernstein, Wang, Azizzadenesheli, et al. | 2018 | arXiv:1802.04434 | Elementwise geometry | yes |
| Old Optimizer, New Norm: An Anthology | Bernstein, Newhouse | 2024 | arXiv:2409.20325 | Optimizer = norm dictionary | yes |
| Modular Duality in Deep Learning | Bernstein, Newhouse | 2024 | arXiv:2410.21265 | Duality view | yes |
| Scalable Optimization in the Modular Norm | Large, Liu, Huh, et al. | 2024 | arXiv:2405.14813 | Modular norm | yes |
| Training Deep Learning Models with Norm-Constrained LMOs (Scion) | Pethick, Xie, Antonakopoulos, et al. | 2025 | arXiv:2502.07529 | Scion | yes |
| Shampoo: Preconditioned Stochastic Tensor Optimization | Gupta, Koren, Singer | 2018 | arXiv:1802.09568 | Background | yes |
| SOAP: Improving and Stabilizing Shampoo using Adam | Vyas, Morwani, Zhao, et al. | 2024 | arXiv:2409.11321 | Background | yes |
| Optimizing Neural Networks with Kronecker-factored Approximate Curvature | Martens, Grosse | 2015 | arXiv:1503.05671 | Background | yes |
| Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer | Yang, Hu, Babuschkin, et al. | 2022 | arXiv:2203.03466 | µP (H6) | yes |
| Large Batch Training of Convolutional Networks (LARS) | You, Gitman, Ginsburg | 2017 | arXiv:1708.03888 | Per-layer trust ratio | yes |
| Large Batch Optimization for Deep Learning: Training BERT in 76 minutes (LAMB) | You, Li, Reddi, et al. | 2019 | arXiv:1904.00962 | Per-layer trust ratio | yes |
| Why Transformers Need Adam: A Hessian Perspective | Zhang, Chen, Ding, et al. | 2024 | arXiv:2402.16788 | Block-Hessian heterogeneity | yes |
| Adam-mini: Use Fewer Learning Rates To Gain More | Zhang, Chen, Li, et al. | 2024 | arXiv:2406.16793 | H9. Abstract: "50% less memory"; partitioning by Hessian structure removes ≥99.9% of v's learning rates. | yes (abstract read) |
| Heavy-Tailed Class Imbalance and Why Adam Outperforms Gradient Descent on Language Models | Kunstner, Yadav, Milligan, et al. | 2024 | arXiv:2402.19449 | H5 mechanism | yes |
| Muon is Scalable for LLM Training (Moonlight) | Liu, Su, Yao, et al. | 2025 | arXiv:2502.16982 | Muon RMS-matching | yes |
| Kimi K2: Open Agentic Intelligence | Kimi Team | 2025 | arXiv:2507.20534 | MuonClip / QK-clip (§2.2 consistency check). Abstract: QK-clip "to address training instability". | yes (abstract read) |
| Dion: Distributed Orthonormalized Updates | Ahn, Xu, Abreu, et al. | 2025 | arXiv:2504.05295 | Block orthogonalization for communication | yes |
| MuonBP: Faster Muon via Block-Periodic Orthogonalization | Khaled, Ozkara, Yu, Hong, Park | 2025 | arXiv:2510.16981 (found by title search; the PDF gives no ID) | Block orthogonalization | yes |
| VeLO: Training Versatile Learned Optimizers by Scaling Up | Metz, Harrison, Freeman, et al. | 2022 | arXiv:2211.09760 | Learned optimizers | yes |
| Symbolic Discovery of Optimization Algorithms (Lion) | Chen, Liang, Huang, et al. | 2023 | arXiv:2302.06675 | Lion | yes |
| Descending through a Crowded Valley — Benchmarking Deep Learning Optimizers | Schmidt, Schneider, Hennig | 2020 | arXiv:2007.01547 | Tuning-budget protocol | yes |
| Benchmarking Neural Network Training Algorithms (AlgoPerf) | Dahl, Schneider, Nado, et al. | 2023 | arXiv:2306.07179 | Tuning-budget protocol | yes |

## B2. Literature study for the experimental phase (added 2026-09-23)

| Title | Authors | Year | Identifier | Relevance | Claim used / checked | Content check |
|---|---|---|---|---|---|---|
| **From SGD to Muon: Adaptive Optimization via Schatten-p Norms** | Massena, Friedrich, Serrurier | 2026 (19 May) | arXiv:2605.19781 | **Direct prior work on MOG's central idea.** It picks the LMO geometry per layer, dynamically, from gradient and activation statistics, using a closed-form random-feature-regression surrogate. The design space runs from SGD to Muon (Schatten-p, i.e. PDF's α axis) plus preconditioning, and recovers SGD, Muon, Adam and MuAdam. Overhead is ~3%. It matches or beats the better of Muon and AdamW in 3 settings. | This contradicts the PDF §2.4 gap claim that "nobody selects the geometry per block from measurement" (THEORY D10). MOG's remaining distinctions are the grouping axis 𝒢 (heads/experts/rows), the oracle-vs-proxy validation methodology, and the damping/shrinkage theory. | Abstract read |
| Noise-Adaptive Layerwise Learning Rates: Accelerating Geometry-Aware Optimization for Deep Neural Network Training | Hao, Gong, Xu, Wang, Liu | 2025 (rev. 2026) | arXiv:2510.14009 | Adapts *per-layer LR* within geometry-aware optimizers from gradient-variance estimates in the LMO's dual norm. Related to Prop. 4's noise-driven scale. | Background | Abstract read |
| AdaMuon: Adaptive Muon Optimizer | (see arXiv) | 2025 | arXiv:2507.11005 | Element-wise second moment after orthogonalization. Close to NorMuon, which we implement. | Background | Title and ID from search listing only |
| Scion (norm-constrained LMOs), Rec. 3.1 | Pethick et al. | 2025 | arXiv:2502.07529 | Our `scion` arm | Input/embedding: ColNorm (per token); hidden: Spectral √(d_out/d_in); output: **Sign** (1/d_in); biases: RMS. This differs from the PDF §2.2 prescription of per-token rows on the output layer, and our `duality` vs `scion` arms test exactly that difference. | HTML (summarized) |
| Adam-mini partition principle | Zhang et al. | 2024 | arXiv:2406.16793 | Our `adam_mini` arm | Q/K by heads, V/attn.proj/MLP by output neurons, embed/output by tokens; v_b = EMA of mean(g⊙g) over the block. | HTML (summarized) |
| NorMuon algorithm | Li et al. | 2025 | arXiv:2510.05491 | Our `normuon` rule | M←β1 M+(1−β1)G; O=NS5(M); v←β2 v+(1−β2) mean_cols(O⊙O); Ô=O/(√v+ε); step 0.2·η·√(mn)/‖Ô‖_F; β1=β2=0.95. | HTML (summarized) |
| Lion update rule | Chen et al. | 2023 | arXiv:2302.06675 | Our `lion` rule | c=β1 m+(1−β1)g; update sign(c); m←β2 m+(1−β2)g; β1=0.9, β2=0.99. Taken from the paper's algorithm, which is well known; the abstract does not state it. | Background knowledge; ID verified |
| Moonlight RMS matching | Liu et al. | 2025 | arXiv:2502.16982 | The 0.2-RMS convention for our LMO rules | Abstract: "carefully adjusting the per-parameter update scale". The 0.2·√max(A,B) factor is from the paper body. | Abstract read; factor from background knowledge (NorMuon's HTML also uses 0.2 RMS) |

## C. Claims in the PDF with no source

- "Gradient spectra are widely reported to become more low-rank as training proceeds" (§4.3). No citation is given, so it is **unverified**. H2's k-shrinkage prediction depends on it, and Exp 1 will measure it directly.

## D. External ideas introduced by MOG (not in the PDF)

- The quintic NS in Muon is approximate. The error of the NS-based nuclear-norm proxy is measured in Exp 0. The source is our own analysis plus the reference code.
- The secant estimate upper-bounds the directional Rayleigh quotient and is measured along the incumbent's step (THEORY.md §5). The source is our own derivation.
