"""Oracle geometry assignment (Exp 2).

The runnable implementation is experiments/03_oracle/oracle.py: single-block switch,
h-step rollouts from a frozen checkpoint on a shared data stream, and an alternate-stream
stability check. It moves here behind a model-agnostic
`oracle(checkpoint, loss_fn, data_stream, candidates, horizon)` API after Stage 3
(docs/FRAMEWORK.md, step 2). It is not moved now because Stage 3 runs are executing it.
"""

import numpy as np


def spearman(a, b) -> float:
    """Rank correlation used to score proxies against the oracle (ties broken by order)."""
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])
