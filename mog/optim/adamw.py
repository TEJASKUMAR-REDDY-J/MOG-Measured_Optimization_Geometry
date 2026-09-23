"""AdamW baseline.

`torch.optim.AdamW` is the reference implementation; this module pins the project
defaults. The per-block `adam` rule in blockwise.py is tested against `torch.optim.Adam`
with the same betas/eps (tests/test_blockwise.py).
"""

import torch

BETAS, EPS = (0.9, 0.95), 1e-8


def adamw(params, lr: float, weight_decay: float = 0.0) -> torch.optim.AdamW:
    return torch.optim.AdamW(params, lr=lr, betas=BETAS, eps=EPS, weight_decay=weight_decay)
