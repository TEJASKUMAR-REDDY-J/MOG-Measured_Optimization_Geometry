# Exp 2: oracle and proxy validation

`oracle.py` works from a frozen checkpoint. For each 2-D block and each candidate rule, it switches that one block, trains 20 steps on a fixed data stream at the checkpoint LR, and measures the change in validation loss against an all-incumbent rollout. The best candidate is re-checked on a second data stream.

It then scores five proxies against this ground truth: the free E_c, the same-batch E_c, the own-direction E_c, gradient-fit alone, and a one-step lookahead.

| Config | Stage | Evidence class |
|---|---|---|
| `configs/tiny/exp012_oracle_dryrun.yaml` | 2 | pilot |
| `configs/1m/exp200_oracle.yaml` | 3 | measured |
