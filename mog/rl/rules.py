"""JAX port of the BlockOptimizer rules used in the RL paradigm (pivot Phase 2).

Semantics match mog/optim/blockwise.py (cross-checked in tests/test_rl_rules.py):
  every block keeps v <- 0.95 v + 0.05 g^2 (bias-corrected by 1 - 0.95^t);
  adam / adam_rms:  m <- 0.9 m + 0.1 g, d = m_hat / (sqrt(v_hat) + 1e-8)  (adam_rms: then RMS 0.2)
  LMO rules:        m <- 0.95 m + 0.05 g, u = 0.05 g + 0.95 m, d = lmo(u) rescaled to RMS 0.2
The rule of each 2-D block is a traced integer (lax.switch), so the oracle can switch rules
without recompiling. 1-D blocks always use adam.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

RULES = ("adam", "adam_rms", "spectral", "polar_svd", "randspec", "sign", "rows", "kaon")
RULE_ID = {r: i for i, r in enumerate(RULES)}
NS = (3.4445, -4.7750, 2.0315)
RMS = 0.2


def newton_schulz(G, steps=5):
    a, b, c = NS
    tall = G.shape[0] > G.shape[1]
    X = G.T if tall else G
    X = X / (jnp.linalg.norm(X) + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if tall else X


def _svd_map(u, f):
    U, S, Vh = jnp.linalg.svd(u, full_matrices=False)
    return (U * f(S, u)) @ Vh


def _kaon(S, u):
    x = S / jnp.maximum(jnp.linalg.norm(u), 1e-30)
    for _ in range(5):
        x = 4.1 * x * (1 - x * x) ** 2
    return x


def _rms(d):
    return d * (RMS * jnp.sqrt(d.size) / jnp.maximum(jnp.linalg.norm(d), 1e-30))


def direction(rule_id, g, m, v_hat, t, key):
    """Returns (d, m_new) for a 2-D block."""
    bc1 = 1 - 0.9 ** t

    def adam(_):
        mn = 0.9 * m + 0.1 * g
        return mn / bc1 / (jnp.sqrt(v_hat) + 1e-8), mn

    def lmo(f):
        def run(_):
            mn = 0.95 * m + 0.05 * g
            return _rms(f(0.05 * g + 0.95 * mn)), mn
        return run

    branches = [
        adam,
        lambda _: (lambda d, mn: (_rms(d), mn))(*adam(None)),
        lmo(newton_schulz),
        lmo(lambda u: _svd_map(u, lambda S, _: jnp.ones_like(S))),
        lmo(lambda u: _svd_map(u, lambda S, _: jax.random.uniform(key, S.shape))),
        lmo(jnp.sign),
        lmo(lambda u: u / jnp.maximum(jnp.linalg.norm(u, axis=1, keepdims=True), 1e-30)),
        lmo(lambda u: _svd_map(u, _kaon)),
    ]
    return jax.lax.switch(rule_id, branches, None)


def init(params):
    z = jax.tree.map(jnp.zeros_like, params)
    return {"m": z, "v": jax.tree.map(jnp.zeros_like, params), "t": jnp.zeros((), jnp.int32)}


def step(params, grads, state, rule_ids, lr, key):
    """params/grads: flat dict name -> array; rule_ids: dict name -> int32 (2-D blocks only)."""
    t = state["t"] + 1
    tf = t.astype(jnp.float32)
    new_p, new_m, new_v = {}, {}, {}
    for i, name in enumerate(sorted(params)):
        g, m = grads[name], state["m"][name]
        v = 0.95 * state["v"][name] + 0.05 * g * g
        v_hat = v / (1 - 0.95 ** tf)
        if g.ndim == 2:
            d, m = direction(rule_ids[name], g, m, v_hat, tf, jax.random.fold_in(key, i))
        else:
            m = 0.9 * m + 0.1 * g
            d = m / (1 - 0.9 ** tf) / (jnp.sqrt(v_hat) + 1e-8)
        new_p[name] = params[name] - lr * d
        new_m[name], new_v[name] = m, v
    return new_p, {"m": new_m, "v": new_v, "t": t}
