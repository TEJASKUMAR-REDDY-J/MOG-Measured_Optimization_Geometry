# THEORY

Source: `granular_optimizer_geometry.pdf` (the "PDF"), the project's primary
specification. That document says of itself that it is a background study with
no training experiments. Every equation here comes from the PDF unless it is
tagged **[MOG-derived]**, meaning we derived or checked it ourselves, or
**[external]**, meaning it comes from another source. Unit tests that check a
statement are listed as `tests/...`.

Status keys: **proved in PDF** means the PDF gives a derivation. **checked** means we
re-derived it or tested it numerically. **unverified** means we have not yet
checked it independently.

---

## 1. An optimizer is a choice of norm (PDF §2.1)

Take a parameter block $W\in\mathbb R^{m\times n}$ with momentum $B$, which estimates $\nabla_W f$, and a norm $\|\cdot\|$. Then

$$\mathrm{lmo}(B)=\arg\max_{\|\Delta\|\le 1}\langle B,\Delta\rangle,\qquad \langle B,\mathrm{lmo}(B)\rangle=\|B\|_*,\qquad W\leftarrow W-\eta\,\mathrm{lmo}(B).$$

| geometry | primal $\|\Delta\|$ | dual $\|B\|_*$ | lmo$(B)$ | $C=\sup\|\Delta\|_F^2/\|\Delta\|^2$ | code |
|---|---|---|---|---|---|
| Euclidean | $\|\Delta\|_F$ | $\|B\|_F$ | $B/\|B\|_F$ | 1 | `Euclidean` |
| elementwise | $\max\|\Delta_{ij}\|$ | $\sum\|B_{ij}\|$ | sign$(B)$ | $mn$ | `Elementwise` |
| per-neuron rows | $\max_i\|\Delta_{i,:}\|_2$ | $\sum_i\|B_{i,:}\|_2$ | rows / row norm | $m$ | `RowNorm("rows")` |
| per-token cols | $\max_j\|\Delta_{:,j}\|_2$ | $\sum_j\|B_{:,j}\|_2$ | cols / col norm | $n$ | `RowNorm("cols")` |
| grouped spectral ($k$ rows) | $\max_g\|\Delta_g\|_{2\to2}$ | $\sum_g\|B_g\|_{\rm nuc}$ | blockwise $U_gV_g^\top$ | $\frac mk\min(k,n)$ | `Spectral(k=k)` |
| full spectral | $\|\Delta\|_{2\to2}$ | $\|B\|_{\rm nuc}$ | $UV^\top$ | $\min(m,n)$ | `Spectral()` |

Status: the LMO/duality identities, unit norm, and the $C$ values (sup property and attainment) are **checked** in `tests/test_geometry.py`.

Correspondences (PDF): momentum SGD is the Euclidean row. signSGD is the elementwise row. AdamW is a *smoothed* elementwise method, and its EMA second moment, ε, bias correction and decoupled weight decay are real differences from sign descent. Muon is the full-spectral row, with $UV^\top$ approximated by Newton–Schulz.

**[MOG-derived] caveat:** Muon's quintic Newton–Schulz coefficients (3.4445, −4.7750, 2.0315) do not converge to $UV^\top$. The output's singular values fall roughly in [0.7, 1.2]. So Muon is only approximately the full-spectral LMO, and $\langle B,\mathrm{NS}(B)\rangle$ is only approximately $\|B\|_{\rm nuc}$. `tests/test_geometry.py::test_newton_schulz_approximates_polar_factor` checks this. Exp 0 measures the size of the error.

## 2. The (G, α) design space (PDF §3.1)

Split a tensor into groups $\mathcal G=\{g\}$ and let $B_g=U_g\Sigma_gV_g^\top$. The PDF defines

$$\Delta W_g=-\eta\,\lambda_g\,s_g\,\frac{U_g\Sigma_g^\alpha V_g^\top}{Z_g(\alpha)},\qquad Z_g(\alpha)=\|U_g\Sigma_g^\alpha V_g^\top\|_{\mathcal G},\qquad \alpha\in[0,1].$$

Here α=1 is Euclidean (Schatten-2) and α=0 is spectral (Schatten-∞). In general the constraint is $\|\Delta\|_{S_p}\le1$ with $\alpha=1/(p-1)$. The PDF also states $U\Sigma^\alpha V^\top=(BB^\top)^{(\alpha-1)/2}B$.

**[MOG-derived]** closed forms used in `Spectral(alpha=...)`. Let $p=1+1/\alpha$, $q=1+\alpha$ and $r=\min(k,n)$:
- lmo$_g$ $=U_g\Sigma_g^\alpha V_g^\top/\|\sigma_g\|_q^\alpha$ (unit $S_p$ norm)
- dual norm $=\sum_g\|B_g\|_{S_q}$
- $C=|\mathcal G|\cdot r^{(1-\alpha)/(1+\alpha)}$

Status: **checked**, including the special cases k=1 → rows, k=m → full, α=1 → Euclidean, and the $(BB^\top)^{(\alpha-1)/2}B$ identity (`tests/test_geometry.py`).

Corners (PDF): whole tensor with α=1 is momentum SGD. 1×1 with α=0 is signSGD ≈ AdamW. 1×n rows with α=0 is the per-neuron update. k×n with α=0 is per-head, per-expert or grouped Muon. Whole tensor with α=0 is Muon/Scion, and adding a per-row $s_g$ gives NorMuon. Whole tensor with α∈(0,1) is partial whitening, which is unexplored.

Implementation status: α>0 is supported only through an exact SVD. The PDF says fractional α can be computed with Muon-style polynomial iterations. That route is **not implemented**.

## 3. Symmetry principle (PDF §3.2), an unverified design heuristic

Each geometry is equivariant under a different group:
- full spectral: $W\mapsto QWR^\top$
- grouped spectral: block-diagonal rotations
- rows: permutations and sign flips of output units
- elementwise: permutations and sign flips of both bases

The PDF proposes this rule: *choose $\mathcal G$ as the coarsest partition that the architecture's symmetry group still preserves.* That gives per-head blocks for Q/K/V/O, per-expert blocks in MoE, per-token rows for the embedding and head, row groups before a nonlinearity, and full spectral only for weights that act on the basis-agnostic residual stream. This is a **hypothesis** (H4). It is not a theorem.

## 4. Descent efficiency and its decomposition (PDF §4.1–4.2)

**Prop. 1.** Let $L=\sup_\Delta\langle\Delta,H\Delta\rangle/\|\Delta\|^2$ be the curvature in the chosen norm. With step $\Delta=-t\,\mathrm{lmo}(B)$:

$$f(W+\Delta)-f(W)\le -t\|B\|_*+\tfrac12t^2L,\qquad t^\star=\|B\|_*/L,\qquad \text{gain}=\|B\|_*^2/(2L).$$

**Corollary.** Write $L=L_F\,C\,a$ with curvature-fit $a\in(0,1]$ and gradient-fit $\theta=\|B\|_*^2/\|B\|_F^2$. Then

$$\frac{\text{gain}(c)}{\text{gain(Euclid)}}=\frac{\theta_c/C_c}{a_c},\qquad \theta_c/C_c\le1\ \text{(Cauchy–Schwarz; equality iff lmo}(B)\parallel B).$$

Special cases:
- $\theta_{\rm spec}/C_{\rm spec}=r_{\rm eff}/\min(m,n)$
- $\theta_{\rm row}/C_{\rm row}=m_{\rm eff}/m$
- $\theta_{\max}/C_{\max}=d_{\rm eff}/(mn)$

Status: the Cauchy–Schwarz bound, the equality case and the effective-dimension identities are **checked** (`tests/test_selection_metrics.py`). $a\le1$ follows from $L\le\lambda_{\max}(H)\sup\|\Delta\|_F^2/\|\Delta\|^2$, and we **checked** that derivation.

The PDF's central interpretive claim is that adaptive methods buy curvature-fit ($a_c\ll1$) and pay for it in gradient-fit. Within this bound framework the claim is a consequence of the corollary. Whether it explains real training is **unverified**. That is H1 and H2.

**[MOG-derived] limitations of Prop. 1 as a selection rule:**
1. Prop. 1 compares *guaranteed-decrease bounds*, not realized decreases. Because $L$ is a sup over all directions, the bound can be loose in very different ways for different geometries.
2. Computing $L$ exactly under non-Euclidean norms is hard in general (for example, $\ell_\infty$ Rayleigh quotients). The method therefore uses the secant proxy of §5 instead, which is a different quantity.
3. It is a one-step criterion. The PDF's own "greedy myopia" risk (§8) applies.

## 5. The selection signal is claimed to be nearly free (PDF §4.4, Prop. 2)

- $\|B\|_*=\langle B,\mathrm{lmo}(B)\rangle$, so each candidate's dual norm costs one LMO. $\|B\|_1$ and $\sum_i\|B_{i,:}\|$ are $O(mn)$, and the PDF says $\|B\|_{\rm nuc}=\langle B,\mathrm{NS}(B)\rangle$ comes free with Muon.
- Secant curvature: $\hat L_c^{(t)}=\|G_t-G_{t-1}\|_{*,c}\,/\,\|\Delta_{t-1}\|_c$, smoothed with an EMA. [external] This is the "trajectory smoothness" estimate that Gluon plots (arXiv:2505.13416). We checked the formula in its HTML text.
- Selection score: $E_c=\theta_c/\hat L_c$.

**[MOG-derived] caveats. These are central risks for H2.**
1. On a quadratic, $\langle\Delta,H\Delta\rangle\le\|H\Delta\|_*\|\Delta\|$. So $\hat L_c$ is an **upper bound on the directional Rayleigh quotient along the step taken**. It is not the sup-curvature $L_c$ of Prop. 1. Checked in `tests/test_curvature.py`.
2. $\Delta_{t-1}$ is the step the *incumbent* geometry took. For every other candidate, $\hat L_c$ measures curvature along a direction that candidate would not take. The proxy may therefore be biased toward the incumbent, or in some other systematic way. The oracle experiment (Exp 2) is the test of this.
3. Under isotropic curvature, the secant taken along a geometry's own flat LMO direction recovers $L_F C_c$ exactly ($a_c=1$). This is checked, and it confirms that the scaling convention matches the corollary.
4. $\langle B,\mathrm{NS}(B)\rangle\neq\|B\|_{\rm nuc}$ (see §1), so the "free" spectral dual norm is biased. A smoke pipeline check (64×64, 2 seeds, *not a result*) showed a relative error of about −0.14 to −0.31, depending on the spectrum. Exp 0 quantifies it at 512×512.
5. Whether a secant needs "no extra backward pass" depends on having $G_{t-1}$ and $G_t$ on the same minibatch. With different minibatches the difference contains gradient noise that is not curvature. This is **unverified** and needs measuring in Exp 1.

## 6. Global damping (PDF §4.5, Prop. 3)

Use a local quadratic with blocks ℓ, unit LMO directions $u_\ell$, $M_{\ell k}=\langle u_\ell,H_{\ell k}u_k\rangle$, diagonal steps $t_\ell=\|g_\ell\|_*/M_{\ell\ell}$ and a global multiplier γ. Let $\tilde M=D^{-1/2}MD^{-1/2}$ and $S=\sum_\ell\|g_\ell\|_*^2/M_{\ell\ell}$. Then

$$\text{decrease}\ge\gamma S-\tfrac12\gamma^2\lambda_{\max}(\tilde M)S,\qquad\gamma^\star=1/\lambda_{\max}(\tilde M),\qquad\text{decrease}=S/(2\lambda_{\max}(\tilde M)).$$

Status: **checked by hand**. The key step is $v^\top Mv=w^\top\tilde Mw\le\lambda_{\max}\|w\|^2=\lambda_{\max}S$ with $w=D^{1/2}v$. $\lambda_{\max}(\tilde M)\ge1$ because $\tilde M$ has a unit diagonal. The derivation needs $M\succeq0$ and $M_{\ell\ell}>0$, which a nonconvex loss can violate. **Not implemented yet.** It becomes relevant in Stage C.

## 7. Granularity as a bias–variance trade (PDF §4.6, Prop. 4)

Model: $\theta_g=\theta_0+\delta_g$ with $\delta_g\sim\mathcal N(0,\tau^2)$, and observations $\hat\theta_g=\theta_g+e_g$ with $e_g\sim\mathcal N(0,s_g^2)$. The James–Stein posterior is $\theta_g^{\rm post}=\theta_0+\lambda_g(\hat\theta_g-\theta_0)$ with $\lambda_g=\tau^2/(\tau^2+s_g^2)$. The plug-in estimate is $\hat\lambda=\max(0,1-\hat s^2/\hat V)$.

The PDF says $s_g^2\propto1/(\text{group size}\times\text{batch}\times\text{SNR}^2)$. It predicts $k_{\min}\propto1/B_{\rm tok}$ (H3) and places NorMuon at λ=1 and Adam-mini at λ=0.

**[MOG-derived]** reading of Fig. 2a: $s^2=1/(k\cdot\text{fan\_in}\cdot\nu^2)$ reproduces the PDF's k=1 values (≈0.79, 0.25, 0.036 for ν=0.1, 0.03, 0.01). Fig. 2b cannot be reproduced exactly because the PDF gives no per-token SNR values. Status: a model, **unverified** on real gradients. Not implemented.

## 8. Width scaling (PDF §4.7, Prop. 5)

Alignment $\chi^2=\frac1{d_{\rm out}}\sum_i\cos^2\angle(\Delta W_{i,:},x)$ and $c^\star=\Theta(1/(\sqrt{d_{\rm in}}\chi))$. The PDF claims this recovers the RMS-matching factor for spectral geometry, µP's $1/d_{\rm in}$ for elementwise geometry, and exponents in [−1/2, 0] for rows. Status: **unverified**, and not re-derived yet (H6).

## 9. Cost (PDF §4.8, Prop. 6)

Newton–Schulz on k-row groups costs ≈ 4Tkmn FLOPs. The overhead is $\frac{2T}{3}\frac{k}{B_{\rm tok}}$, which is about 2.6% at k=4096 and 512k tokens per step.

**[MOG-derived]:** counting the $k^3$ term, the exact per-iteration cost is $4k^2n+2k^3$ FLOPs, so at k=n the PDF formula undercounts by 1.5×. `Spectral.cost` uses the full count (`tests/test_geometry.py::test_ns_cost_matches_pdf_scaling`). The PDF does not state the layer shape used in Fig. 1d, and Exp 0 assumes 4096×4096.

## 10. Proposed method (PDF §5), spec only, nothing implemented

Per-block state: momentum, EMAs of $\theta_c$ and $\hat L_c$, the incumbent $c_\ell$, $s_g$, λ, $\hat V$ and $\hat s^2$.

- Every $T_{\rm sel}$ (100–500) steps: score $E_c$ and switch only if the argmax beats the incumbent by δ on two consecutive probes.
- Every $T_{\rm damp}$ (~1000) steps: re-estimate γ.
- Weight decay should match the geometry, i.e. a pull toward a radius in $\|\cdot\|_{\mathcal G}$.

## 11. Discrepancies found in the PDF

| # | location | issue |
|---|---|---|
| D1 | §2.3 bullet 3 | Links depth-localization to "(H7)". In the §6 table it is **H8**, and H7 is the LR ceiling. |
| D2 | §2.3 bullet 2 | "bounds gain below neuron granularity (see H8)". The matching hypothesis is **H9**. |
| D3 | §2.1, §10 | Gluon "(ICML 2026)": the arXiv listing and HTML show no venue. **Unverified.** |
| D4 | §2.3 | Attributes to arXiv:2606.04662 a *layer-by-layer decomposition of the gap* where the majority comes from boundary layers. The abstract instead emphasizes within-layer curvature. The HTML full text (read through a summarizing fetch, so it needs a manual re-read) places a "concentrated in the first and deepest layers" statement in Appendix C, and that statement is about the *within-layer sharpness (NDS) gap*, not the loss gap. The claim is **partially supported and overstated**. |
| D5 | §2.3 | Adam-mini memory "~45–50%". The abstract says "50% less memory". This is minor. |
| D6 | §4.3 | "Gradient spectra are widely reported to become more low-rank as training proceeds" has no citation. **Unverified.** H2's k-shrinkage prediction depends on it. |
| D7 | §4.3 | The r_eff column and full-spectral column are closed-form. We checked them: 512, 281.65, 28.28, 5.30 and 1.000, 0.5501, 0.0552, 0.0104. The printed values match to their rounding. The rows and elementwise columns are Monte Carlo, and Exp 0 checks them. |
| D8 | §4.4 | Treats $\langle B,\mathrm{NS}(B)\rangle$ as exactly $\|B\|_{\rm nuc}$. Muon's NS is approximate (see §1 and §5.4). |
| D9 | §4.4 | Calls the secant "the curvature" of Prop. 1. It is a different, directional, upper-bound quantity measured along the incumbent's step (see §5). |
| D10 | §2.4 | "Nobody selects the geometry per block from measurement." **Contradicted** by arXiv:2605.19781 (Massena et al., May 2026), which selects an LMO geometry per layer from runtime gradient and activation statistics. The PDF cites June 2026 papers, so this work predates it. Novelty has to rest on the grouping axis, the oracle/proxy validation, and Props. 3–4 (LITERATURE.md B2). |
