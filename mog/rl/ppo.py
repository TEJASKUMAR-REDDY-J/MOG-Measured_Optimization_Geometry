"""PPO on gymnax MinAtar, fully jitted (PureJaxRL structure), with per-block MOG rules.

Separate actor and critic MLPs: in (obs -> W, boundary), h (W -> W, hidden), out (W -> A or 1,
boundary). Weights are stored (out, in) like torch, so `rows` = per output neuron.
Defaults follow PureJaxRL's MinAtar PPO except width (256, per the proposal's R5 repair).

cfg: env, total_steps, num_envs 64, num_steps 128, epochs 4, minibatches 8, gamma .99, lam .95,
clip .2, ent .01, vf .5, max_grad_norm .5, width 256, lr, anneal (linear to 0), rules (spec),
seed, eval_envs 512, eval_cap 1000.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import gymnax
from mog.rl import rules as R

DEFAULTS = {"num_envs": 64, "num_steps": 128, "epochs": 4, "minibatches": 8, "gamma": 0.99, "lam": 0.95, "clip": 0.2,
            "ent": 0.01, "vf": 0.5, "max_grad_norm": 0.5, "width": 256, "anneal": True, "eval_envs": 512, "eval_cap": 1000}


def blocks_meta():
    meta = {}
    for net in ("actor", "critic"):
        meta[f"{net}.in"] = {"kind": f"{net}_in", "role": "boundary"}
        meta[f"{net}.h"] = {"kind": f"{net}_h", "role": "hidden"}
        meta[f"{net}.out"] = {"kind": f"{net}_out", "role": "boundary"}
        for k in ("in", "h", "out"):
            meta[f"{net}.{k}_b"] = {"kind": "gain", "role": "gain"}
    return meta


def resolve(spec: dict) -> dict:
    """Rule spec (same keys as configs/arms.yaml: name > kind > role > matrix > default) -> ids."""
    out = {}
    for name, m in blocks_meta().items():
        if m["role"] == "gain":
            continue
        rule = next(spec[k] for k in (name, m["kind"], m["role"], "matrix", "default") if k in spec)
        out[name] = R.RULE_ID[rule]
    return out


class PPO:
    def __init__(self, cfg: dict):
        self.c = DEFAULTS | cfg
        self.env, self.env_params = gymnax.make(self.c["env"])
        self.obs_dim = int(np.prod(self.env.observation_space(self.env_params).shape))
        self.n_act = self.env.action_space(self.env_params).n
        c = self.c
        self.batch = c["num_envs"] * c["num_steps"]
        self.n_updates = c["total_steps"] // self.batch
        self.update = jax.jit(self._update)
        self.evaluate = jax.jit(self._evaluate)

    # ---- model
    def init_params(self, key):
        W, orth = self.c["width"], jax.nn.initializers.orthogonal
        p, ks = {}, jax.random.split(key, 6)
        for j, (net, out, scale) in enumerate((("actor", self.n_act, 0.01), ("critic", 1, 1.0))):
            for k, (name, shape, s) in enumerate((("in", (W, self.obs_dim), 2**0.5), ("h", (W, W), 2**0.5), ("out", (out, W), scale))):
                p[f"{net}.{name}"] = orth(s)(jax.random.fold_in(ks[j], k), shape)
                p[f"{net}.{name}_b"] = jnp.zeros(shape[0])
        return p

    @staticmethod
    def _mlp(p, net, x):
        h = jax.nn.relu(x @ p[f"{net}.in"].T + p[f"{net}.in_b"])
        h = jax.nn.relu(h @ p[f"{net}.h"].T + p[f"{net}.h_b"])
        return h @ p[f"{net}.out"].T + p[f"{net}.out_b"]

    def policy(self, p, obs):
        x = obs.reshape(obs.shape[0], -1).astype(jnp.float32)
        return self._mlp(p, "actor", x), self._mlp(p, "critic", x)[:, 0]

    # ---- state
    def init_state(self, seed: int, rules: dict):
        key = jax.random.PRNGKey(seed)
        kp, ke, kr = jax.random.split(key, 3)
        params = self.init_params(kp)
        obs, env_state = jax.vmap(self.env.reset, (0, None))(jax.random.split(ke, self.c["num_envs"]), self.env_params)
        return {"params": params, "opt": R.init(params), "env_state": env_state, "obs": obs, "rng": kr,
                "ep_ret": jnp.zeros(self.c["num_envs"]), "update": jnp.zeros((), jnp.int32),
                "rule_ids": {k: jnp.int32(v) for k, v in resolve(rules).items()}}

    def lr_at(self, lr, u):
        return lr * (1 - u / self.n_updates) if self.c["anneal"] else lr

    # ---- rollout + GAE
    def _collect(self, s):
        c = self.c

        def env_step(carry, _):
            s = carry
            rng, ka, ks = jax.random.split(s["rng"], 3)
            logits, value = self.policy(s["params"], s["obs"])
            a = jax.random.categorical(ka, logits)
            logp = jax.nn.log_softmax(logits)[jnp.arange(a.shape[0]), a]
            obs, env_state, r, term, trunc, _ = jax.vmap(self.env.step, (0, 0, 0, None))(
                jax.random.split(ks, c["num_envs"]), s["env_state"], a, self.env_params)
            done = jnp.logical_or(term, trunc).astype(jnp.float32)
            ep = s["ep_ret"] + r
            s = s | {"rng": rng, "obs": obs, "env_state": env_state, "ep_ret": jnp.where(done, 0.0, ep)}
            return s, (carry["obs"], a, logp, value, r, done, ep)

        s, (obs, act, logp, val, rew, done, ep) = jax.lax.scan(env_step, s, None, c["num_steps"])
        _, last_v = self.policy(s["params"], s["obs"])

        def gae(carry, x):
            adv, nv = carry
            d, v, r = x
            delta = r + c["gamma"] * nv * (1 - d) - v
            adv = delta + c["gamma"] * c["lam"] * (1 - d) * adv
            return (adv, v), adv

        _, adv = jax.lax.scan(gae, (jnp.zeros_like(last_v), last_v), (done, val, rew), reverse=True)
        flat = lambda x: x.reshape((self.batch,) + x.shape[2:])
        data = tuple(map(flat, (obs, act, logp, val, adv, adv + val)))
        return s, data, done, ep

    def loss(self, p, mb, eps=None):
        """PPO loss on minibatch mb; eps {block: (N, out)} is added to that block's pre-activation
        (zero-valued probes whose gradient is the per-sample output gradient)."""
        c = self.c
        o, a, lp_old, v_old, A, G = mb
        x = o.reshape(o.shape[0], -1).astype(jnp.float32)
        e = eps or {}

        def mlp(net):
            h = jax.nn.relu(x @ p[f"{net}.in"].T + p[f"{net}.in_b"] + e.get(f"{net}.in", 0))
            h = jax.nn.relu(h @ p[f"{net}.h"].T + p[f"{net}.h_b"] + e.get(f"{net}.h", 0))
            return h @ p[f"{net}.out"].T + p[f"{net}.out_b"] + e.get(f"{net}.out", 0)

        logits, v = mlp("actor"), mlp("critic")[:, 0]
        lsm = jax.nn.log_softmax(logits)
        lp = lsm[jnp.arange(a.shape[0]), a]
        A = (A - A.mean()) / (A.std() + 1e-8)
        ratio = jnp.exp(lp - lp_old)
        pg = -jnp.minimum(ratio * A, jnp.clip(ratio, 1 - c["clip"], 1 + c["clip"]) * A).mean()
        vc = v_old + jnp.clip(v - v_old, -c["clip"], c["clip"])
        vl = 0.5 * jnp.maximum((v - G) ** 2, (vc - G) ** 2).mean()
        ent = -(jnp.exp(lsm) * lsm).sum(-1).mean()
        return pg + c["vf"] * vl - c["ent"] * ent

    # ---- one PPO update (rollout + epochs); lr_override < 0 -> schedule
    def _update(self, s, lr, lr_override):
        c = self.c
        s, data, done, ep = self._collect(s)
        lr_now = jnp.where(lr_override >= 0, lr_override, self.lr_at(lr, s["update"]))

        def epoch(carry, key):
            p, opt, k2 = carry
            perm = jax.random.permutation(key, self.batch)
            mbs = jax.tree.map(lambda x: x[perm].reshape((c["minibatches"], -1) + x.shape[1:]), data)

            def mb_step(carry, mb):
                p, opt, k2 = carry
                g = jax.grad(self.loss)(p, mb)
                gn = jnp.sqrt(sum(jnp.sum(x * x) for x in jax.tree.leaves(g)))
                g = jax.tree.map(lambda x: x * jnp.minimum(1.0, c["max_grad_norm"] / (gn + 1e-6)), g)
                k2, ko = jax.random.split(k2)
                p, opt = R.step(p, g, opt, s["rule_ids"], lr_now, ko)
                return (p, opt, k2), None

            return jax.lax.scan(mb_step, (p, opt, k2), mbs)[0], None

        rng, ke, ko = jax.random.split(s["rng"], 3)
        (p, opt, _), _ = jax.lax.scan(epoch, (s["params"], s["opt"], ko), jax.random.split(ke, c["epochs"]))
        n_done = done.sum()
        train_ret = jnp.where(n_done > 0, (ep * done).sum() / jnp.maximum(n_done, 1), jnp.nan)
        return s | {"params": p, "opt": opt, "rng": rng, "update": s["update"] + 1}, train_ret

    # ---- evaluation: first episode of each of eval_envs envs, fixed key -> paired across candidates
    def _evaluate(self, params, key):
        c, n = self.c, self.c["eval_envs"]
        k0, k1 = jax.random.split(key)
        obs, st = jax.vmap(self.env.reset, (0, None))(jax.random.split(k0, n), self.env_params)

        def f(carry, k):
            obs, st, ret, alive = carry
            ka, ks = jax.random.split(k)
            logits, _ = self.policy(params, obs)
            a = jax.random.categorical(ka, logits)
            obs, st, r, term, trunc, _ = jax.vmap(self.env.step, (0, 0, 0, None))(jax.random.split(ks, n), st, a, self.env_params)
            ret = ret + r * alive
            return (obs, st, ret, alive * (1 - jnp.logical_or(term, trunc))), None

        (_, _, ret, alive), _ = jax.lax.scan(f, (obs, st, jnp.zeros(n), jnp.ones(n)), jax.random.split(k1, c["eval_cap"]))
        return ret.mean(), alive.mean()
