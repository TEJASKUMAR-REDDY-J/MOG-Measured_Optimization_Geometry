"""BlockOptimizer for non-GPT models (pivot Phase 2).

Blocks carry a `role`: "hidden" (interior weight matrices / conv kernels), "boundary"
(input and output layers) or "gain" (1-D biases and norm gains, always adam).
Rule specs resolve by priority: block name > kind > role > "matrix" (any >=2-D) > default,
so the arms in configs/arms.yaml ({hidden: spectral, default: adam}, ...) apply unchanged.
Conv kernels (out, in, kh, kw) are updated as out x (in*kh*kw) matrices (Muon convention).
"""

from __future__ import annotations

import torch

from mog.optim.blockwise import LMO, BlockOptimizer


def resolve_task_rules(blocks: list[dict], spec: dict) -> dict[str, str]:
    out = {}
    for b in blocks:
        keys = [b["name"], b["kind"], b["role"]] + (["matrix"] if b["param"].ndim >= 2 else []) + ["default"]
        rule = next(spec[k] for k in keys if k in spec)
        if b["param"].ndim < 2 and rule in LMO:
            rule = "adam"  # LMO rules are undefined on 1-D gains
        out[b["name"]] = rule
    return out


class TaskBlockOptimizer(BlockOptimizer):
    def direction(self, block, g, st, rule):
        if g.ndim <= 2:
            return super().direction(block, g, st, rule)
        shape = g.shape
        flat = {k: (v.view(shape[0], -1) if torch.is_tensor(v) and v.shape == shape else v) for k, v in st.items()}
        d = super().direction(block | {"param": block["param"].view(shape[0], -1)}, g.reshape(shape[0], -1), flat, rule)
        for k, v in flat.items():  # scalars (t) and new entries (e.g. normuon's vrow) go back to st
            if not (torch.is_tensor(v) and torch.is_tensor(st.get(k)) and st[k].shape == shape):
                st[k] = v
        return d.view(shape)
