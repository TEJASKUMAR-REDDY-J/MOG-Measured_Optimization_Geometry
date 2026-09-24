# MOG from zero: what this project is, what we are building, and why we run each experiment

This document is for a reader who has never studied neural networks, language models, or optimizers. It starts from nothing and builds up everything needed to understand the MOG project: its question, its experiments, its results, and what it is ultimately trying to build. Technical terms are explained when they first appear and collected again in the glossary at the end.

The more technical companion documents are:
- [THEORY.md](THEORY.md): the equations
- [RESEARCH_PLAN.md](RESEARCH_PLAN.md): the staged plan
- [HYPOTHESES.md](HYPOTHESES.md): the formal hypothesis table
- [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md): the dated record of every run
- [RESULTS.md](RESULTS.md): the measured numbers

---

## Part 1. The world this project lives in

### 1.1 What a neural network is, with no jargon

A neural network is a **very large, adjustable calculation**. Numbers go in, the calculation runs, and numbers come out.

- For an image classifier, the numbers going in are pixel brightnesses. The numbers coming out are scores such as "how much this looks like a cat".
- For a language model, the numbers going in represent pieces of text. The numbers coming out are scores for what the next piece of text should be.

The calculation has thousands to billions of adjustable knobs called **parameters** (also called **weights**). At the start, the knobs are set randomly and the network produces garbage. **Training** is the process of turning the knobs, a tiny bit at a time, until the outputs become useful.

A useful mental picture is a **mixing desk** in a recording studio with millions of sliders. You cannot set each slider by hand: there are too many, and nobody knows the right positions. Instead, you use an automatic procedure that listens to how wrong the output sounds and nudges every slider slightly in a direction that makes it sound less wrong. Then it repeats, millions of times.

### 1.2 Matrices: how the knobs are organised

The knobs are not a loose pile. They come in **blocks**, and most blocks are rectangular grids of numbers called **matrices**. A matrix with 128 rows and 512 columns holds 65,536 knobs.

A matrix is a machine that turns one list of numbers into another list of numbers. Feed in a list of 512 numbers, and each of the 128 rows combines those 512 inputs into one output. Each row behaves like one "neuron": it takes all the inputs, weighs them, and produces one number.

This organisation matters a great deal for this project. A matrix is not just 65,536 independent knobs. It has **rows** (neurons), **columns** (inputs), and an overall **shape of action**: it stretches some directions of input and squashes others. Different ways of adjusting the knobs respect that structure to different degrees, and that is exactly what this project studies.

### 1.3 Training: loss, gradient, and taking steps

To train, you need three things.

1. **A loss.** One number that says how wrong the network currently is. For a language model it measures how surprised the model was by the real next piece of text. Lower is better. Our losses are reported in "nats" (a unit of surprise). A drop of 0.35 nats is a large improvement, and 0.01 is a small one.

2. **A gradient.** For every knob, the gradient answers one question: *if I nudge this knob up slightly, does the loss go up or down, and how fast?* Computing it for all knobs at once is called the **backward pass** (or backpropagation). It costs about twice as much as running the network forward.

3. **A step.** Move every knob a little in the direction that reduces the loss. The size of the move is controlled by the **learning rate (LR)**. If it is too small, training crawls. If it is too large, training overshoots and becomes unstable (it "diverges").

The classic picture is a **hiker in thick fog on a hilly landscape**, trying to reach the lowest valley. The hiker cannot see the valley and can only feel the slope underfoot (the gradient). Each step goes downhill. The landscape has an astronomically large number of directions, one per knob.

Two refinements appear everywhere:

- **Mini-batches and noise.** The true loss is defined over all the training data, but computing it exactly every step is too expensive. So each step uses a small random **batch** of examples. The slope felt from a batch is a noisy estimate of the true slope, a bit like feeling the ground through a thick boot.
- **Momentum.** Instead of following each step's noisy slope directly, keep a running average of recent slopes and follow that. This smooths the noise, the way a heavy ball rolling downhill ignores small bumps. In the code this running average is called the **momentum buffer**. The project's formulas call it $B$.

### 1.4 Language models and transformers

A **language model** reads text and predicts what comes next. Text is chopped into **tokens**, which can be single characters or common word pieces. The model's only job during training is: *given the tokens so far, give high probability to the actual next token.* Large modern chat systems are huge versions of this. The project uses tiny versions, because the question is about *how to train*, and small models answer that faster and cheaper.

The architecture used almost universally is the **transformer**. Our small transformer has these blocks, and each one matters later because each block may want a different optimizer:

| Block | What it does | Shape in our 0.82M-parameter model |
|---|---|---|
| **Embedding** (`embed`) | A lookup table: each token gets its own row, a list of 128 numbers representing it | one row per token type × 128 |
| **Positional embedding** (`pos`) | A lookup table giving each position in the text its own list of numbers | 128 positions × 128 |
| **Attention: q, k, v, o** | Lets each position "look at" earlier positions. q (query) and k (key) decide *where* to look, v (value) decides *what* to take, and o (output) mixes the result back in. The work is split into 4 independent **heads**, each looking for a different kind of relationship. | 128 × 128 each |
| **MLP: up, down** | A per-position "thinking" step. `up` widens 128 numbers to 512, a nonlinearity is applied, and `down` compresses back to 128. | 512 × 128 and 128 × 512 |
| **LayerNorm gains** | Small lists of numbers that rescale values | 128 numbers each |
| **Head** (`head`, the unembedding) | Turns the final 128 numbers into one score per possible next token | one row per token type × 128 |

Attention and MLP together form one **layer**. Our model stacks 4 layers. Everything flows through a shared channel called the **residual stream**: each block reads from it and adds its contribution back in.

### 1.5 Optimizers: the rules for turning the knobs

An **optimizer** is the precise rule that turns "here is the gradient (or momentum)" into "here is how much to move each knob". This choice matters enormously. With the same model, the same data and the same compute, different optimizers produce very different results. The ones in this project:

- **SGD with momentum.** Move each knob proportionally to its (averaged) slope. It is the simplest rule, and it is weak on language models, because some knobs have huge slopes and others tiny ones, so no single step size suits them all.
- **Adam / AdamW.** Gives *every individual knob its own automatic step size*, by dividing the slope by a running estimate of how large that knob's slopes usually are. Knobs with habitually tiny slopes get boosted, and knobs with huge slopes get calmed. It is the industry default. (AdamW is Adam with a particular way of shrinking weights over time. We set that shrinking to zero, so here the two are the same.)
- **signSGD / Signum.** Look only at the *sign* of each knob's slope, up or down, and move every knob by the same fixed amount. It is a crude cousin of Adam.
- **Lion.** A sign-based optimizer found by an automated search, with a particular way of mixing old and new momentum.
- **Adam-mini.** Adam, but sharing one step size across a whole group of knobs (for example a whole neuron or a whole head) instead of one per knob. It saves memory.
- **Muon.** The newcomer. It treats a matrix as a **whole object** instead of a bag of knobs. It takes the matrix of slopes and "orthogonalizes" it: it keeps the *directions* the matrix wants to change in, but equalises how strongly each direction changes. The effect is a well-balanced update of the whole matrix. Muon has trained language models about twice as efficiently as Adam in public reports. It is computed with a short, fixed recipe of matrix multiplications called **Newton–Schulz iteration**.
- **NorMuon.** Muon plus a per-neuron rebalancing step afterwards.

The standard modern practice is a **hand-written two-bucket rule**: use Muon for the big internal matrices (attention and MLP), and use AdamW for everything else (embeddings, head, gains). Nobody derived that rule from first principles or measured whether it is right block by block. It became the default because it works.

---

## Part 2. The idea behind MOG

### 2.1 An optimizer is a choice of ruler

The central insight the project builds on, from existing research, is this:

> Each optimizer is "take the steepest downhill step", but each one measures the *size* of a step with a different ruler.

"Steepest downhill step of size 1" depends on what "size 1" means. There are several natural choices:

- **Elementwise ruler.** A step's size is its *largest single knob change*. The steepest step under this ruler moves *every* knob by the same amount in its downhill direction. That is the sign rule, the ancestor of Adam.
- **Per-row ruler.** A step's size is the *largest change to any one neuron* (row), measured as the ordinary length of that row's change. The steepest step normalises each row separately: every neuron gets a same-sized update, pointing wherever its own slope points.
- **Grouped ruler.** Same idea, but for groups of rows, for example all the rows belonging to one attention head.
- **Whole-matrix (spectral) ruler.** A step's size is the *strongest stretching effect the change has on any input*. The steepest step under this ruler is exactly Muon's orthogonalized update.
- **Euclidean ruler.** The ordinary straight-line length of the whole change. The steepest step is plain normalized gradient descent.

So "Adam or Muon?" is really "**which ruler should measure steps for this block?**" A ruler encodes an assumption about which knob changes are equivalent, big, or dangerous for that block. There is no reason one assumption should be right for every block: an embedding table, an attention head and an MLP projection do very different jobs.

### 2.2 The design space: granularity × whitening

The source document (the PDF this project started from) turns this into a two-dial design space:

1. **Granularity**: how big are the groups of knobs that share a ruler? Options run from a single knob, to a row, to k rows, to a head, to the whole matrix.
2. **Whitening strength (α)**: within a group, how much do you equalise the directions? Full equalisation (α=0) is Muon-style; none (α=1) is plain gradient.

Known optimizers sit at corners of this map:

| Corner | Optimizer |
|---|---|
| Single knob, full equalisation | sign / Adam-like |
| Row | per-neuron normalisation |
| Whole matrix, full equalisation | Muon |
| Whole matrix, none | normalized SGD |

Most of the interior of the map, such as "per-head Muon" or "half whitening (α=½)", has barely been tried.

### 2.3 The MOG question

**MOG = Measured Optimization Geometry.** The project asks:

1. **Is the best ruler really different for different blocks?** (Heterogeneity.)
2. **Can we tell which ruler a block wants, cheaply, while training?** (Measurement.)
3. **Does acting on that knowledge train better models for the same budget?** (Payoff.)

The PDF proposes a specific measurement. For each block and each candidate ruler, compute a **score** $E_c$ = (how well the ruler's steepest step lines up with the slope) ÷ (how curved the landscape is when measured with that ruler). The ruler with the highest score should give the biggest immediate drop in loss. The two ingredients are:

- **Gradient-fit (θ).** How much of the slope's downhill power the ruler's step captures. It is cheap to compute.
- **Curvature (L̂).** How fast the slope itself changes as you move. The PDF proposes estimating it "for free" by comparing the slopes at two consecutive training steps (a **secant** estimate): how much the slope changed, divided by how far the knobs moved.

The PDF's most striking theoretical point is that **Adam and Muon are worse aligned with the slope than plain gradient descent**. They win only because they choose directions where the landscape is gentle (low curvature), and that lets them take bigger safe steps. It is a trade: give up alignment, gain curvature. MOG's bet is that the best trade differs by block.

### 2.4 What we ultimately want to build

Two things, in order of importance.

1. **A measurement framework (the durable goal).** A software package that can attach to any training run and answer, per block of any network:
   - what the update geometry looks like (a **geometry profiler**, which we call the "atlas");
   - which ruler each block *actually* benefits from, via a controlled test (the **oracle harness**);
   - whether any cheap signal predicts that answer (**proxy validation**).

   This is useful even if no new optimizer comes out of it, because it replaces convention with measurement. The design is in [FRAMEWORK.md](FRAMEWORK.md).

2. **An adaptive optimizer (the optional goal).** One that picks the ruler per block while training. It gets built **only if** the measurements show it can work. The project deliberately refuses to build it first and then look for evidence. That order is the main safeguard against fooling ourselves.

---

## Part 3. How we test ideas without fooling ourselves

Training experiments are noisy, and it is very easy to "find" an improvement that is really luck, extra tuning effort, or a cherry-picked number. The project uses these safeguards:

| Safeguard | What it means | Why |
|---|---|---|
| **Staged ladder** | Start with pure maths, then tiny models, then ~1M-parameter models, and only then larger. Scaling up requires a written GO decision. | Compute is expensive. Most ideas fail cheaply at small scale, and scale should answer a question, not decorate a result. |
| **Pre-registration** | Before each real experiment, write down the hypothesis, the metric and the exact pass/fail rule in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md). | Stops the goalposts moving after seeing the data. |
| **Matched budget** | Every optimizer sees exactly the same number of tokens (pieces of text) and steps. | Otherwise "better" might just mean "trained longer". |
| **Equal tuning budget** | Every optimizer gets exactly one tuned setting (its learning rate), tried on the same 3-point grid. | Optimizer comparisons are notoriously unfair when one method gets more tuning. |
| **Separate seeds** | A **seed** fixes the random starting knobs and the order of the data. Tuning uses seed 0, and every reported comparison uses seeds 1, 2 and 3. | Stops tuning luck leaking into the result. Differences are reported with 95% confidence intervals across seeds. |
| **Negative results count** | A failed hypothesis is written up with the same care as a success. | The honest null result is a real finding. |
| **Reproducibility** | Every run saves its config, software versions, git commit, the exact command, logs and machine-readable results in its own never-overwritten folder under `results/raw/`. | Anyone can re-run and check. |

A **confidence interval** such as "+0.015 [−0.001, +0.031]" reads as follows: the average difference was +0.015, and allowing for seed-to-seed noise, the true difference is plausibly anywhere from −0.001 to +0.031. If the interval contains 0, we cannot claim a difference.

---

## Part 4. The experiments, what each one tests, and why

All training runs use a small transformer on the complete works of Shakespeare (about 1 MB of text), on a laptop CPU. The main ("Stage 3") model has **0.82 million parameters**. That is tiny by industry standards, which is deliberate: a test at this size is cheap enough to repeat properly.

### Stage 0: does the maths hold? (Exp 0, synthetic)

- **What.** No training. We build artificial slope matrices whose "shape" we control, from well-spread (all directions equally strong) to very lopsided (a few directions dominate, called *low-rank*). For each, we compute how well each ruler's step lines up with the slope. We then compare against the table and figures printed in the PDF.
- **Why.** If the project's basic arithmetic were wrong, every later experiment would be built on sand.
- **Result.** Every number in the PDF's table reproduced within random variation. We also found something the PDF assumed without checking: Muon's fast Newton–Schulz recipe does **not** exactly produce the ideal whole-matrix step. Its "free" measurement is off by 2–22%, and for lopsided matrices its direction is quite far from ideal (similarity 0.63 out of 1). That undercuts the PDF's claim that part of the score comes "for free".

### Stage 2: does the machinery work? (exp010–012, tiny 0.11M model)

- **What.** Fourteen optimizers were each trained at 5 learning rates and 2 seeds for 300 steps, followed by a dry run of the measurement tools.
- **Why.** To make sure every optimizer is implemented correctly and trains stably, and to find sensible learning-rate ranges before spending hours at larger scale. No scientific claims are made at this stage.
- **Result.** All 14 optimizers trained stably. As a correctness check, our Muon implementation matches the reference implementation **bit for bit**.

### Stage 3a: fair baselines (exp150 tuning → exp160 comparison)

- **What.** The same 14 optimizers on the 0.82M model. Each gets a 3-point learning-rate search on seed 0, then its best setting is trained on seeds 1–3.
- **Why.** Any claim that "per-block choice is better" needs strong, fairly tuned opponents.
- **Result.**
  - The biggest effect in the whole project: **Muon-style (whole-matrix) rulers on the internal matrices beat Adam by about 0.35 nats.** That is a large, clear gap, consistent with published reports.
  - Within the Muon family (Muon, NorMuon, and the PDF's "duality" assignment) the differences are within noise.
  - Adam, sign-based rules and per-row rules all trail far behind. Plain SGD is worst.

### Stage 3b: the atlas, measuring during training (exp100)

- **What.** While training (under Muon and under Adam), every 50 steps and for every block we record:
  - how lopsided its momentum is (**effective rank**);
  - how well each ruler lines up with it;
  - the curvature estimates;
  - how noisy its slope is (**signal-to-noise ratio**);
  - which ruler the PDF's score would choose.
- **Why.** Before asking "which ruler is best", check whether the ingredients the PDF relies on behave as it assumes.
- **Result.** Three surprises:
  1. The PDF assumed slopes become *more* lopsided over training. They became *less* lopsided in every block.
  2. The "free" curvature estimate, from two consecutive steps on different data, came out 2–4 times larger than the proper same-data estimate. It mostly measures batch-to-batch noise, not curvature.
  3. The PDF's score picked the sign ruler in about 90% of cases, yet sign-based optimizers lose badly in practice.

### Stage 3c: the oracle, ground truth per block (exp200)

- **What.** We save the model at three moments in training (steps 100, 300 and 500). For each of the 27 matrices and each of 6–9 candidate rulers, we switch *only that one matrix* to the candidate ruler, train 20 more steps on identical data, and measure the loss. The ruler that helps most is the **oracle's choice** for that block at that moment. We then re-run the winner on different data to check the win is not a fluke.
- **Why.** This is the honest answer to "which ruler does this block want?", measured directly. It is expensive, but it is ground truth.
- **Result.**
  - In 60 of 81 cases, the oracle preferred something other than the standard recipe. 36 of those held up on fresh data.
  - The gains are tiny, about 0.00007 nats per block.
  - The most consistent preference: **NorMuon for the MLP "up" matrices**, in all 12 cases.

### Stage 3d: does the cheap score predict the oracle? (H2, same run)

- **What.** For every block, we compare the ranking of rulers by the PDF's score with their ranking by the oracle.
- **Why.** An adaptive optimizer is only feasible if a cheap signal predicts the expensive ground truth.
- **Result.**
  - **The PDF's score fails.** Its rankings are slightly *backwards* (correlation −0.33, where +1 is perfect and 0 is random), and its top pick matches the oracle only 8.6% of the time, below the 12% expected by pure chance.
  - The "same-data" variant also fails.
  - A variant that measures curvature **along each candidate's own step** works well (correlation 0.76, top pick right 48% of the time), but it costs an extra backward pass per candidate, so it is not free.
  - Two further lessons. Rulers that line up best with the slope do *worse* (this is the PDF's own trade-off, confirmed). And the best one-step move is *not* the best 20-step move ("greedy myopia").

### Stage 3e: does following the oracle beat the best optimizer? (H1, exp290 + exp300)

- **What.** We train a model that follows the oracle's per-block choices (switching at steps 100, 300 and 500), tuned and seeded exactly like the baselines.
- **Why.** This is the payoff question. If even perfect knowledge of per-block preferences does not help, no cheap selector will.
- **Result.** **No.** It scored 1.705 against 1.690 for the best baseline (NorMuon): a difference of +0.015, with interval [−0.001, +0.031]. At best that is a tie and it leans worse. The small per-block wins, measured one block at a time, do not add up when applied together.

### Gate decision after Stage 3

Scaling up the PDF's method is **not justified** by the evidence. Instead, four redesign experiments target the specific open questions.

### Redesign experiments (done 2026-09-24)

| ID | Question | Why it matters | Pre-registered pass rule | Verdict |
|---|---|---|---|---|
| exp400 + exp410 (H2′) | Does the *own-direction* score, which worked in exp200, still predict the oracle on a **fresh model and fresh data**? | exp200's success for this score was discovered after the fact. It has to be re-tested cleanly before anyone builds on it. | Correlation ≥ 0.5 **and** top-pick agreement significantly above chance | **PASS.** Correlation 0.77; top pick right 42% of the time vs 12% by chance. The PDF's free score failed again. |
| exp420 (additivity) | Do single-block wins add up when combined, and do they last beyond 20 steps (tested at 100)? | Explains *why* the oracle-following model failed: interacting blocks, or wins that fade. | Combined gain > 0 and ≥ half the sum of the single gains | **Mostly yes.** Wins combine at 2 of 3 checkpoints, but they are tiny, often fade by 100 steps, and some do not repeat on new data. |
| exp430 + exp431 (H4 control) | Is splitting attention matrices **by head** better than splitting them into random groups of the same size? | Per-head splitting lost to whole-matrix Muon. This checks whether the head structure matters at all, or whether splitting simply hurts. | Per-head beats random with an interval excluding 0 | **FAIL.** Per-head vs random groups: −0.024 [−0.060, +0.012]. Both are worse than whole-matrix Muon. |
| exp440 (H5) | For the embedding and head tables, is per-token normalisation better than Adam once there are **realistic rare tokens**? | The PDF predicts rare words benefit. Our character-level text has only 65 symbols, so this needed a word-piece tokenizer with about 11,700 token types. | Overall gain with an interval excluding 0 **and** a bigger gain on the rarest tokens | **FAIL.** Per-token normalisation is *worse* than Adam (+0.052), including on rare tokens. |

---

## Part 5. What the results mean in plain terms

1. **The ruler matters a lot, but mainly in one place.** Switching the big internal matrices from Adam-style to Muon-style rulers is worth about 0.35 nats, which is huge. Choosing between the good whole-matrix rulers barely matters.
2. **The standard two-bucket recipe is hard to beat at this size.** Fine-grained per-block choices, even with perfect hindsight, did not improve on it.
3. **The PDF's cheap selector does not work as specified.** Its curvature estimate is dominated by noise, and its score points the wrong way. This is a clear, pre-registered negative result.
4. **One promising lead survived its re-test.** Measuring curvature along each candidate's own proposed step predicted the ground truth well, and it did so again on a fresh model and fresh data (H2′). It costs one extra backward pass per candidate.
6. **Other appealing ideas from the PDF failed cleanly.** Splitting attention by head is no better than random splitting (H4). Per-token updates for the vocabulary tables are worse than Adam even with realistic rare tokens (H5). The oracle's per-block wins are real but tiny and short-lived, which is why following them did not beat the best optimizer.
5. **Novelty caveat.** While this project ran, we found that another group (arXiv:2605.19781, May 2026) had already published a data-driven per-layer choice of ruler, and CMuon (arXiv:2608.02502) had already used group-wise Muon. So MOG's contribution must lean on its **measurement methodology** (the atlas, the oracle, and pre-registered proxy validation) rather than on "we invented per-block selection".

These results come from **one small model, one dataset, and a CPU budget**. They may change at larger scale or on other tasks. They are the honest answer at this scale, and exactly the kind of answer the staged plan exists to produce before large compute is spent.

---

## Part 6. What comes next

- If H2′ holds, design a pre-registered test of an adaptive optimizer that uses the own-direction score, with its extra cost counted honestly.
- Extract the atlas and oracle into a general-purpose package ([FRAMEWORK.md](FRAMEWORK.md)).
- Questions beyond language models (reinforcement learning, self-supervised learning, image models, diffusion) are written up as a separate project in [PROBLEM_STATEMENT_CROSS_DOMAIN.md](PROBLEM_STATEMENT_CROSS_DOMAIN.md). They change the problem too much to fold in here.

---

## Glossary

| Term | Meaning |
|---|---|
| **Parameter / weight / knob** | An adjustable number inside the network |
| **Block** | A group of parameters with one role, e.g. one matrix such as `L2.up` (layer 2, MLP up-projection) |
| **Matrix** | A rectangular grid of numbers. Rows ≈ neurons, columns ≈ inputs. |
| **Loss** | One number measuring how wrong the model is. Lower is better. In nats here. |
| **Validation loss** | Loss on text the model never trained on. It measures real skill, not memorisation. |
| **Gradient** | For each parameter, how the loss changes if that parameter is nudged |
| **Backward pass** | The computation that produces all gradients (≈ 2× the cost of running the model) |
| **Learning rate (LR)** | How big each training step is |
| **Batch** | The small random sample of training text used for one step. It makes the gradient noisy. |
| **Momentum** | A running average of recent gradients that smooths out noise |
| **Optimizer** | The rule turning gradients or momentum into parameter changes |
| **Token** | A unit of text: a character, or a common word piece |
| **Transformer** | The standard architecture of language models: attention plus MLP layers |
| **Attention head** | One of several parallel "look-back" channels inside attention |
| **Embedding / head (unembedding)** | Lookup tables mapping tokens in and scores out |
| **Norm / ruler / geometry** | The way the size of a step is measured. It determines which step counts as "steepest". |
| **LMO (linear minimization oracle)** | The rule that finds the steepest step under a given ruler |
| **Orthogonalization / Newton–Schulz** | Muon's way of equalising all directions of a matrix update, via a short fixed recipe |
| **Effective rank** | How many directions a matrix meaningfully uses. Low means "lopsided". |
| **Gradient-fit (θ)** | How well a ruler's steepest step lines up with the gradient |
| **Curvature / secant** | How fast the slope changes as you move. The secant estimates it from two gradients. |
| **Proxy** | A cheap signal used to predict something expensive |
| **Oracle** | The expensive ground truth: try each option directly and see which helps |
| **Seed** | The number that fixes all randomness in a run (starting weights, data order) |
| **Pre-registration** | Writing the pass/fail rule down before running the experiment |
| **Confidence interval (95% CI)** | The range where the true difference plausibly lies. If it contains 0, there is no claim. |
| **Spearman ρ** | A rank correlation: +1 means identical ordering, 0 unrelated, −1 reversed |
| **Pilot / smoke run** | A small run that checks the machinery works. It is never cited as evidence. |

## Where to find things in the repository

| Path | Contents |
|---|---|
| `mog/geometry/` | The rulers: elementwise, rows, grouped, spectral |
| `mog/optim/` | The optimizers, including the per-block optimizer that can switch rulers mid-training |
| `mog/selection/` | Measurements: gradient-fit, curvature, atlas metrics, hysteresis |
| `experiments/` | One folder per experiment family. `sweep.py` runs optimizer comparisons. |
| `configs/` | One file per experiment. Each file fully determines its run. |
| `results/raw/<experiment>/<timestamp>/` | Every run's config, logs, numbers and exact reproduce command |
| `docs/` | This document and the technical companions |
