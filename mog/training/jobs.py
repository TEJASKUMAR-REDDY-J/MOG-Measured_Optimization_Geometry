"""Picklable jobs for process pools, dispatching on cfg["task"]:
absent -> GPT (mog.training.trainer); "rl" -> JAX PPO (mog.rl.train); else a torch paradigm task."""

from __future__ import annotations

import torch

_CKPT: dict = {}
KEEP = ("history", "final_val", "final_train", "diverged", "wall_s", "final_metrics")


def run_train(cfg: dict, run_dir=None) -> dict:
    task = cfg.get("task")
    if task is None:
        from mog.training.trainer import train
        r = train(cfg | {"threads": cfg.get("threads", 1)}, run_dir=run_dir, log=lambda *a: None)
    elif task == "rl":
        from mog.rl.train import train_rl
        r = train_rl(cfg, run_dir=run_dir)
    else:
        from mog.training.task_trainer import train_task
        r = train_task(cfg, run_dir=run_dir, log=lambda *a: None, final_extra=True)
    return {k: r[k] for k in KEEP if k in r}


def _load(path):
    if path not in _CKPT:
        _CKPT.clear()
        if path.endswith(".pkl"):
            from mog.rl.train import load_ckpt
            _CKPT[path] = load_ckpt(path)
        else:
            _CKPT[path] = torch.load(path, weights_only=False)
    return _CKPT[path]


def run_rollout(ckpt_path: str, switches: dict, horizon: int, data_seed: int) -> float:
    """From a checkpoint, switch {block: rule}, train `horizon` steps (updates for RL) at the
    checkpoint's LR held constant on stream `data_seed`; return the task's val loss."""
    torch.set_num_threads(1)
    ck = _load(ckpt_path)
    task = ck["cfg"].get("task")
    if task is None:
        from mog.training.trainer import rollout_job
        return rollout_job(ckpt_path, switches, horizon, data_seed)
    if task == "rl":
        from mog.rl.train import rollout_rl
        return rollout_rl(ck, switches, horizon, data_seed)
    from mog.training.task_trainer import train_task
    from mog.training.trainer import lr_at
    cfg = ck["cfg"] | {"steps": ck["step"] + horizon, "data_seed": data_seed, "eval_every": 10**9,
                       "const_lr": lr_at(ck["step"] - 1, ck["cfg"]), "checkpoints": [], "threads": 1}

    def sched(step, opt):
        for block, rule in switches.items():
            opt.set_rule(block, rule)

    return train_task(cfg, start=ck, rule_schedule=sched if switches else None, log=lambda *a: None)["final_val"]


def blocks_meta(ck) -> dict:
    """{name: {kind, role}}; GPT checkpoints predate the field, so derive it from block names."""
    if "blocks_meta" in ck:
        return ck["blocks_meta"]
    from mog.optim.blockwise import HIDDEN
    meta = {}
    for n in ck["opt"]["rules"]:
        kind = n.split(".")[-1]
        role = "hidden" if kind in HIDDEN else "boundary" if kind in ("embed", "pos", "head") else "gain"
        meta[n] = {"kind": kind, "role": role}
    return meta


def run_probe(ckpt_path: str, K: int, probe_seed: int) -> dict:
    """Block statistics (mog/selection/block_probe.py; RL: mog/rl/probe.py) at a checkpoint,
    on K independent batches from stream probe_seed (never used for training)."""
    torch.set_num_threads(1)
    ck = _load(ckpt_path)
    cfg, task = ck["cfg"], ck["cfg"].get("task")
    if task == "rl":
        from mog.rl.probe import probe_rl
        return probe_rl(ck, K, probe_seed)
    from mog.selection.block_probe import probe
    if task is None:
        from mog.training.trainer import build, get_data
        data = get_data(cfg.get("data", "char"))
        model, bl, opt = build(cfg, data)
        B, T = cfg["batch_size"], cfg["model"]["ctx"]

        def loss_at(k):
            x, y = data.batch("train", k, B, T, probe_seed)
            return model(x, y)
    else:
        from mog.training.task_trainer import build_task
        t, model, bl, opt = build_task(cfg)

        def loss_at(k):
            return t.loss(model, k, probe_seed)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    return probe(model, bl, opt.state, loss_at, K)
