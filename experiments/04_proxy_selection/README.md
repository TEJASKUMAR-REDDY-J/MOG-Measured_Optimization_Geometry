# Proxy selection

Proxy-versus-oracle scoring runs inside Exp 2 (`experiments/03_oracle/oracle.py`). It uses the same checkpoints, so both are computed together. The per-proxy Spearman ρ, top-1 agreement and regret are in the `summary.proxies` field of the exp200 `results.json`.

This directory will hold proxy-only studies, such as EMA smoothing and probe cadence, if H2 is not falsified.
