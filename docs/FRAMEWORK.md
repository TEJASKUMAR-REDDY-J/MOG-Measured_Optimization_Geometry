# MOG as a framework (design proposal)

Automatically deciding each block's geometry is one possible payoff. Whether it works depends on H2, the proxy-validity hypothesis. The framework is useful **even if automatic selection fails**, because the hard, reusable parts are measuring and testing geometry per block, not the decision rule.

## Four layers, each usable alone

| Layer | What it gives a user | Status in repo |
|---|---|---|
| **1. Geometry library** | One interface (`lmo`, `norm`, `dual_norm`, `C`, `cost`) for elementwise, row/column, grouped/per-head/per-expert spectral, partial whitening, and Euclidean geometries. | `mog/geometry` (done, tested) |
| **2. Per-block optimizer** | Assign any rule to any parameter block by name, kind or group. A block can be switched mid-run while keeping its state. One tuned LR is shared across rules through RMS matching. | `mog/optim/blockwise.py` (done; block discovery is transformer-specific) |
| **3. Geometry profiler (atlas)** | A training hook that logs, per block: effective rank/density, gradient-fit per geometry, same-batch secant curvature, gradient SNR and activation participation ratio. It answers "what does my network's gradient geometry look like?" for any model. | `experiments/01_geometry_validation/atlas.py` (works; needs extracting into the library) |
| **4. Oracle harness** | At a checkpoint, for each block and each candidate geometry: a short controlled rollout gives the ground-truth preference. Gives an assignment plan plus a stability check. | `experiments/03_oracle/oracle.py` (works; needs extracting) |
| 5. *(optional)* Online selector | Proxy-scored switching with hysteresis and global damping. Built only if H2 passes. | not built (gated) |

## Proposed public API (sketch)

```python
import mog

blocks = mog.discover(model)                   # Linear/Embedding/Conv/attention heads -> named blocks + kinds
opt = mog.BlockOptimizer(blocks, policy="folklore", lr=3e-3)
#   policy: "folklore" | "duality" | "scion" | dict(kind -> rule) | mog.Plan.load(path)

prof = mog.Profiler(blocks, every=50)          # attach to any training loop
for step, batch in enumerate(loader):
    loss = loss_fn(model, batch); loss.backward()
    prof.observe(step, opt, batch, loss_fn)    # extra passes only on profiling steps
    opt.step(lr_schedule(step))

plan = mog.oracle(checkpoint, loss_fn, data_stream, candidates="default", horizon=20)
plan.save("geometry_plan.json")                # -> policy for the next run
```

## Work needed to make it general (not done; after Stage 3)

1. `mog.discover`: find blocks generically from module types and let users override the head count or grouping. Currently `blocks()` only knows our GPT.
2. Move the atlas and oracle out of the experiment scripts into `mog/profiler.py` and `mog/oracle.py` behind the API above, keeping the experiment scripts as thin callers.
3. A loss-function-agnostic oracle, so RL or self-supervised users supply their own `loss_fn` and data stream. This is the bridge to the cross-domain project.
4. Packaging: docs, examples, and a CPU-sized quickstart.

The online selector (layer 5) is intentionally not part of this plan until the evidence supports it.
