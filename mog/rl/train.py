"""RL training / checkpoint / oracle rollout with the same contract as the torch tasks:
history rows {step, train_loss, val_loss, ...} with val_loss = -(mean eval return), so lower is better.

cfg: task "rl", env, total_steps, lr, rules, seed, eval_every (updates), checkpoints (updates),
plus PPO keys (mog/rl/ppo.py). Checkpoints are pickled numpy pytrees of the full runner state.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mog.rl.ppo import PPO, blocks_meta
from mog.rl.rules import RULE_ID

EVAL_KEY = 1_000_000
_PPO: dict = {}


def get_ppo(cfg):
    keys = ("env", "total_steps", "num_envs", "num_steps", "epochs", "minibatches", "width", "anneal", "eval_envs", "eval_cap",
            "gamma", "lam", "clip", "ent", "vf", "max_grad_norm")
    sub = {k: cfg[k] for k in keys if k in cfg}
    key = tuple(sorted(sub.items()))
    if key not in _PPO:
        _PPO[key] = PPO(sub)
    return _PPO[key]


def evaluate(ppo, params):
    ret, alive = ppo.evaluate(params, jax.random.PRNGKey(EVAL_KEY))
    return {"val": -float(ret), "return": float(ret), "unfinished": float(alive)}


def train_rl(cfg, run_dir: Path | None = None, start: dict | None = None, switches: dict | None = None,
             lr_override: float = -1.0, n_updates: int | None = None, data_seed: int | None = None) -> dict:
    ppo = get_ppo(cfg)
    s = ppo.init_state(cfg["seed"], cfg["rules"]) if start is None else jax.tree.map(jnp.asarray, start["state"])
    if data_seed is not None:  # a different environment/sampling stream from the same state
        s = s | {"rng": jax.random.fold_in(s["rng"], data_seed)}
    if switches:
        s = s | {"rule_ids": s["rule_ids"] | {k: jnp.int32(RULE_ID[r]) for k, r in switches.items()}}
    u0 = int(s["update"])
    u_end = ppo.n_updates if n_updates is None else u0 + n_updates
    ckpts = set(cfg.get("checkpoints", []))
    hist, rets, t0 = [], [], time.time()
    for u in range(u0, u_end):
        s, r = ppo.update(s, cfg["lr"], lr_override)
        rets.append(float(r))
        if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(s["params"])):
            return {"history": hist, "diverged": True, "final_val": float("inf"), "final_train": float("inf"),
                    "wall_s": round(time.time() - t0, 2), "state": None}
        if run_dir is not None and u + 1 in ckpts:
            (run_dir / "checkpoints").mkdir(exist_ok=True)
            with open(run_dir / "checkpoints" / f"step_{u + 1}.pkl", "wb") as f:
                pickle.dump({"step": u + 1, "state": jax.device_get(s), "cfg": cfg, "blocks_meta": blocks_meta(),
                             "lr_at": float(ppo.lr_at(cfg["lr"], u))}, f)
        if (u + 1) % cfg["eval_every"] == 0 or u + 1 == u_end:
            ev = evaluate(ppo, s["params"])
            hist.append({"step": u + 1, "train_loss": -float(np.nanmean(rets[-cfg["eval_every"]:])), "val_loss": ev["val"],
                         "lr": float(ppo.lr_at(cfg["lr"], u)), "wall_s": round(time.time() - t0, 2)})
    return {"history": hist, "diverged": False, "final_val": hist[-1]["val_loss"], "final_train": hist[-1]["train_loss"],
            "wall_s": round(time.time() - t0, 2), "state": s}


def load_ckpt(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def rollout_rl(ck: dict, switches: dict, horizon: int, data_seed: int) -> float:
    r = train_rl(ck["cfg"] | {"eval_every": 10**9}, start=ck, switches=switches, lr_override=ck["lr_at"],
                 n_updates=horizon, data_seed=data_seed)
    return r["final_val"]
