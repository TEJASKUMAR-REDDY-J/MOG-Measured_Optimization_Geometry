"""Steepest descent with momentum under a per-param-group geometry.

    buf <- beta * buf + (1 - beta) * g
    u   <- (1 - beta) * g + beta * buf   (nesterov)   |   buf   (otherwise)
    p   <- (1 - lr * wd) * p - lr * scale(shape) * geometry.lmo(u)

With Spectral(method="ns"), nesterov=True and scale=muon_scale this is the
reference Muon update (verified bit-for-bit in tests/test_muon_equivalence.py).
Any Geometry can be dropped in, so the same geometry code serves as a
standalone optimizer, inside the oracle, and inside the selector.

Weight decay here is decoupled l2 (as in Muon/AdamW). PDF §5 argues decay
should be geometry-matched; that is not implemented yet.
"""

from __future__ import annotations

from typing import Callable

import torch

from mog.geometry import Geometry


def muon_scale(shape: torch.Size) -> float:
    """K. Jordan's Muon shape factor max(1, m/n)^0.5."""
    return max(1.0, shape[-2] / shape[-1]) ** 0.5


class GeometricOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        geometry: Geometry,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        scale: Callable[[torch.Size], float] | None = None,
    ):
        defaults = dict(
            geometry=geometry, lr=lr, momentum=momentum, nesterov=nesterov,
            weight_decay=weight_decay, scale=scale,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta, lr = group["momentum"], group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - beta)
                u = g.lerp(buf, beta) if group["nesterov"] else buf
                d = group["geometry"].lmo(u)
                if group["scale"] is not None:
                    d = d * group["scale"](p.shape)
                p.mul_(1 - lr * group["weight_decay"])
                p.add_(d, alpha=-lr)
        return loss
