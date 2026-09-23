"""GeometricOptimizer(Spectral NS) must reproduce the reference Muon update.

The reference below is copied from K. Jordan's Muon (github.com/KellerJordan/Muon,
muon.py, MIT license) -- zeropower_via_newtonschulz5, muon_update, and the
SingleDeviceMuon parameter step.
"""

import pytest
import torch

from mog.geometry import Spectral
from mog.optim import GeometricOptimizer, muon_scale


def zeropower_via_newtonschulz5(G, steps: int):
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


def reference_step(params, grads, bufs, lr, wd, beta):
    for p, g, buf in zip(params, grads, bufs):
        update = muon_update(g.clone(), buf, beta=beta)
        p.mul_(1 - lr * wd)
        p.add_(update.reshape(p.shape), alpha=-lr)


@pytest.mark.parametrize("shapes,exact", [([(16, 32), (32, 32)], True), ([(48, 16)], False)])
def test_matches_reference_muon(shapes, exact):
    torch.manual_seed(0)
    lr, wd, beta = 0.02, 0.01, 0.95
    ref = [torch.randn(s) for s in shapes]
    ours = [torch.nn.Parameter(p.clone()) for p in ref]
    bufs = [torch.zeros(s) for s in shapes]
    opt = GeometricOptimizer(ours, Spectral(method="ns", ns_dtype=torch.bfloat16),
                             lr=lr, momentum=beta, nesterov=True, weight_decay=wd, scale=muon_scale)
    for _ in range(5):
        grads = [torch.randn(s) for s in shapes]
        reference_step(ref, grads, bufs, lr, wd, beta)
        for p, g in zip(ours, grads):
            p.grad = g.clone()
        opt.step()
    for r, o in zip(ref, ours):
        if exact:  # scale == 1: identical op sequence, must be bit-identical
            assert torch.equal(r, o.data)
        else:      # tall: reference applies the shape factor in bf16, we apply it in fp32
            assert torch.allclose(r, o.data, rtol=0, atol=5e-3)
