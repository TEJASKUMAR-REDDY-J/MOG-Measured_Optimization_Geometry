"""Phase 2 infrastructure: conv blocks update as out x (in*k*k) matrices; role-based rule
resolution; oracle statistics helpers."""

import numpy as np
import torch

from mog.optim.blockwise import BlockOptimizer
from mog.optim.task_optimizer import TaskBlockOptimizer, resolve_task_rules
from mog.selection.oracle import paired_ci, tost


def test_conv_update_equals_flattened_matrix_update():
    torch.manual_seed(0)
    W = torch.randn(8, 3, 3, 3)
    w4, w2 = W.clone(), W.reshape(8, -1).clone()
    o4 = TaskBlockOptimizer([{"name": "c", "param": w4, "kind": "conv", "role": "hidden", "n_head": 1}], {"c": "spectral"})
    o2 = BlockOptimizer([{"name": "c", "param": w2, "kind": "conv", "n_head": 1}], {"c": "spectral"})
    for s in range(3):
        g = torch.randn(8, 3, 3, 3, generator=torch.Generator().manual_seed(s))
        w4.grad, w2.grad = g.clone(), g.reshape(8, -1).clone()
        o4.step(0.01)
        o2.step(0.01)
    assert torch.allclose(w4.reshape(8, -1), w2, atol=1e-6)


def test_resolve_task_rules_by_role():
    bl = [{"name": "a", "param": torch.zeros(4, 4), "kind": "fc", "role": "hidden"},
          {"name": "b", "param": torch.zeros(4, 4), "kind": "head", "role": "boundary"},
          {"name": "c", "param": torch.zeros(4), "kind": "gain", "role": "gain"}]
    r = resolve_task_rules(bl, {"hidden": "spectral", "default": "adam"})
    assert r == {"a": "spectral", "b": "adam", "c": "adam"}
    assert resolve_task_rules(bl, {"matrix": "sign", "default": "sign"})["c"] == "adam"  # LMO never on 1-D


def test_paired_ci_and_tost():
    d = np.array([0.01, 0.012, 0.009, 0.011, 0.010])
    ci = paired_ci(d)
    assert ci["ci95"][0] > 0 and ci["n_pos"] == 5
    assert tost(d - d.mean(), margin=0.005)["equivalent"] and not tost(d, margin=0.005)["equivalent"]
