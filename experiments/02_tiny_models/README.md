# Stage 2: tiny-model pilots

The tiny-model sweep (`exp010_tiny_optimizer_sanity`, config `configs/tiny/exp010_optimizer_sanity.yaml`) uses the shared entry `experiments/sweep.py`. It trains every optimizer arm in `configs/arms.yaml` on a ~0.11M-parameter transformer with a 5-point LR grid, 2 seeds and 300 steps.

Evidence class: pilot. It checks implementation and stability and locates the LR grid centres. It makes no scientific claims.
