"""Training loop for paradigm tasks (mog/paradigms), mirroring mog.training.trainer.train.

cfg keys: task ("supervised" | "ssl" | "flow"), task_cfg {...}, steps, lr, warmup, min_lr_frac,
seed, data_seed, eval_every, rules, optional checkpoints, const_lr, diverge_loss (default 50).
Checkpoints hold model + optimizer state + cfg + blocks_meta {name: {kind, role}}.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import torch

from mog.optim.task_optimizer import TaskBlockOptimizer, resolve_task_rules
from mog.paradigms.fashion import TASKS
from mog.training.trainer import lr_at

_TASK: dict = {}


def get_task(cfg: dict):
    key = (cfg["task"], tuple(sorted(cfg.get("task_cfg", {}).items())))
    if key not in _TASK:
        _TASK[key] = TASKS[cfg["task"]](cfg.get("task_cfg", {}))
    return _TASK[key]


def build_task(cfg: dict):
    task = get_task(cfg)
    model, bl = task.build(cfg["seed"])
    return task, model, bl, TaskBlockOptimizer(bl, resolve_task_rules(bl, cfg["rules"]))


def train_task(cfg: dict, run_dir: Path | None = None, start: dict | None = None,
               rule_schedule: Callable | None = None, callback: Callable | None = None,
               log: Callable = print, final_extra: bool = False) -> dict:
    torch.set_num_threads(cfg.get("threads", 1))
    task, model, bl, opt = build_task(cfg)
    s0 = 0
    if start is not None:
        model.load_state_dict(start["model"])
        opt.load_state_dict(start["opt"])
        s0 = start["step"]
    dseed = cfg.get("data_seed", cfg["seed"])
    ckpts = set(cfg.get("checkpoints", []))
    hist, run_loss, n_loss, t0, diverged = [], 0.0, 0, time.time(), False
    for step in range(s0, cfg["steps"]):
        if rule_schedule is not None:
            rule_schedule(step, opt)
        loss = task.loss(model, step, dseed)
        if not torch.isfinite(loss) or loss.item() > cfg.get("diverge_loss", 50):
            diverged = True
            log(f"diverged at step {step} (loss {loss.item()})")
            break
        for p in model.parameters():
            p.grad = None
        loss.backward()
        if callback is not None:
            callback(step, model, bl, opt)
        opt.step(lr_at(step, cfg))
        run_loss += loss.item()
        n_loss += 1
        if run_dir is not None and step + 1 in ckpts:
            (run_dir / "checkpoints").mkdir(exist_ok=True)
            torch.save({"step": step + 1, "model": model.state_dict(), "opt": opt.state_dict(), "cfg": cfg,
                        "blocks_meta": {b["name"]: {"kind": b["kind"], "role": b["role"]} for b in bl}},
                       run_dir / "checkpoints" / f"step_{step + 1}.pt")
        if (step + 1) % cfg["eval_every"] == 0 or step + 1 == cfg["steps"]:
            ev = task.evaluate(model)
            hist.append({"step": step + 1, "train_loss": run_loss / n_loss, "val_loss": ev["val"],
                         "lr": lr_at(step, cfg), "wall_s": round(time.time() - t0, 2)} | {f"ev_{k}": v for k, v in ev.items() if k != "val"})
            run_loss, n_loss = 0.0, 0
    if not hist and not diverged:  # zero-step rollout: the checkpoint's own loss
        hist.append({"step": s0, "train_loss": float("nan"), "val_loss": task.evaluate(model)["val"], "lr": 0.0, "wall_s": 0.0})
    out = {"history": hist, "diverged": diverged, "wall_s": round(time.time() - t0, 2), "model": model, "opt": opt, "blocks": bl,
           "final_val": float("inf") if diverged else hist[-1]["val_loss"],
           "final_train": float("inf") if diverged else hist[-1]["train_loss"]}
    if final_extra and not diverged:
        out["final_metrics"] = task.evaluate(model, **({"probe": True} if cfg["task"] == "ssl" else {}))
    return out
