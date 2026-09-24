"""RL counterpart of mog/selection/block_probe.probe: the same statistics for the PPO actor/critic
blocks. G_k = gradient of the PPO loss on a fresh rollout k from the checkpoint state
(sampling stream fold_in(probe_seed, k)); Kronecker factors from rollout 0 via zero probes on
each block's pre-activation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import torch

from mog.rl.train import get_ppo
from mog.selection.block_probe import stats_from


def probe_rl(ck: dict, K: int, probe_seed: int) -> dict:
    ppo = get_ppo(ck["cfg"])
    s = jax.tree.map(jnp.asarray, ck["state"])
    names = [n for n in s["params"] if s["params"][n].ndim == 2]
    collect, grad = jax.jit(ppo._collect), jax.jit(jax.grad(ppo.loss))
    G, fac = {n: [] for n in names}, {}
    for k in range(K):
        _, data, _, _ = collect(s | {"rng": jax.random.fold_in(jax.random.PRNGKey(probe_seed), k)})
        g = grad(s["params"], data)
        for n in names:
            G[n].append(torch.tensor(np.asarray(g[n])))
        if k == 0:
            N = data[0].shape[0]
            eps = {n: jnp.zeros((N, s["params"][n].shape[0])) for n in names}
            d = jax.grad(lambda e: ppo.loss(s["params"], data, e))(eps)
            x0 = data[0].reshape(N, -1).astype(jnp.float32)
            for net in ("actor", "critic"):
                h1 = jax.nn.relu(x0 @ s["params"][f"{net}.in"].T + s["params"][f"{net}.in_b"])
                h2 = jax.nn.relu(h1 @ s["params"][f"{net}.h"].T + s["params"][f"{net}.h_b"])
                for layer, x in (("in", x0), ("h", h1), ("out", h2)):
                    n = f"{net}.{layer}"
                    X, D = torch.tensor(np.asarray(x)), torch.tensor(np.asarray(d[n]))
                    fac[n] = (X.T @ X / N, D.T @ D / N)
    m = {n: torch.tensor(np.asarray(s["opt"]["m"][n])) for n in names}
    return stats_from(G, m, fac)
