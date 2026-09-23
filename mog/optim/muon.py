"""Muon = GeometricOptimizer(Spectral NS, Nesterov momentum 0.95, shape factor max(1, m/n)^0.5).

Bit-exact with K. Jordan's reference implementation (tests/test_muon_equivalence.py).
Inside BlockOptimizer the same geometry is the rule "spectral", with RMS-0.2 matching.
"""

import torch

from mog.geometry import Spectral

from .geometric import GeometricOptimizer, muon_scale


def muon(params, lr: float = 0.02, momentum: float = 0.95, weight_decay: float = 0.0,
         ns_steps: int = 5, ns_dtype: torch.dtype = torch.bfloat16) -> GeometricOptimizer:
    return GeometricOptimizer(params, Spectral(method="ns", ns_steps=ns_steps, ns_dtype=ns_dtype),
                              lr=lr, momentum=momentum, nesterov=True, weight_decay=weight_decay,
                              scale=muon_scale)
