"""The JAX rules (mog/rl/rules.py) must reproduce the torch BlockOptimizer update."""

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from mog.optim.blockwise import BlockOptimizer  # noqa: E402
from mog.rl import rules as R  # noqa: E402


@pytest.mark.parametrize("rule", ["adam", "adam_rms", "spectral", "polar_svd", "sign", "rows", "kaon"])
def test_jax_rules_match_torch(rule):
    rng = np.random.default_rng(0)
    W0, b0 = rng.standard_normal((24, 16)).astype(np.float32), rng.standard_normal(24).astype(np.float32)
    tw, tb = torch.tensor(W0.copy()), torch.tensor(b0.copy())
    bl = [{"name": "w", "param": tw, "kind": "x", "n_head": 1}, {"name": "b", "param": tb, "kind": "gain", "n_head": 1}]
    opt = BlockOptimizer(bl, {"w": rule, "b": "adam"})
    p = {"w": jnp.array(W0), "b": jnp.array(b0)}
    st = R.init(p)
    for s in range(4):
        gw, gb = rng.standard_normal((24, 16)).astype(np.float32), rng.standard_normal(24).astype(np.float32)
        tw.grad, tb.grad = torch.tensor(gw), torch.tensor(gb)
        opt.step(0.01)
        p, st = R.step(p, {"w": jnp.array(gw), "b": jnp.array(gb)}, st, {"w": jnp.int32(R.RULE_ID[rule])}, 0.01, jax.random.PRNGKey(s))
    assert np.allclose(np.asarray(p["w"]), tw.numpy(), atol=2e-5)
    assert np.allclose(np.asarray(p["b"]), tb.numpy(), atol=2e-5)
