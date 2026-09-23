# MOG arms (Stage 3+)

- `mog_oracle`: the incumbent Muon-hybrid rules plus the oracle's stable per-block switches, applied piecewise from the checkpoints. Implemented in `experiments/sweep.py` (`oracle_schedule`). Configs: `configs/1m/exp290_mog_lr_tuning.yaml` (LR tuning, seed 0) and `configs/1m/exp300_mog.yaml` (evaluation, seeds 1–3).
- Online proxy selector with hysteresis and damping: **gated** on H2 (see `docs/RESEARCH_PLAN.md` §4). It is not built unless Exp 2 supports the proxy.
