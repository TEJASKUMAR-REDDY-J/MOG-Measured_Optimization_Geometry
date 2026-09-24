"""Training loop shared by every experiment.

cfg keys: model{d,n_layer,n_head,ctx}, batch_size, steps, lr, warmup, min_lr_frac,
seed (init), data_seed (batch stream; defaults to seed), eval_every, eval_batches,
rules (spec for resolve_rules), optional checkpoints [steps], threads.
Checkpoint at step s holds W_s, optimizer state, and (g_{s-1}, Delta_{s-1}) so the
free secant proxy can be computed later.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable

import torch

from mog.data import BPEData, CharData
from mog.models.tiny_transformer import GPT, GPTConfig, blocks
from mog.optim.blockwise import BlockOptimizer, resolve_rules

_DATA: dict[str, CharData] = {}


def get_data(name: str = "char") -> CharData:
    if name not in _DATA:
        _DATA[name] = BPEData() if name == "bpe" else CharData()
    return _DATA[name]


def lr_at(step: int, cfg: dict) -> float:
    if "const_lr" in cfg:  # oracle rollouts hold the checkpoint's LR fixed
        return cfg["const_lr"]
    lr, W, S = cfg["lr"], cfg["warmup"], cfg["steps"]
    if step < W:
        return lr * (step + 1) / W
    frac = (step - W) / max(1, S - W)
    lo = cfg.get("min_lr_frac", 0.1)
    return lr * (lo + (1 - lo) * 0.5 * (1 + math.cos(math.pi * frac)))


def build(cfg: dict, data: CharData):
    torch.manual_seed(cfg["seed"])
    model = GPT(GPTConfig(vocab=data.vocab, **cfg["model"]))
    bl = blocks(model)
    opt = BlockOptimizer(bl, resolve_rules(bl, cfg["rules"]))
    return model, bl, opt


@torch.no_grad()
def evaluate(model, batches) -> float:
    model.eval()
    loss = sum(float(model(x, y)) for x, y in batches) / len(batches)
    model.train()
    return loss


def train(cfg: dict, run_dir: Path | None = None, callback: Callable | None = None,
          start: dict | None = None, rule_schedule: Callable | None = None, log: Callable = print) -> dict:
    """Returns {'history': [...], 'final_val', 'final_train', 'diverged', 'wall_s'}.

    start: a checkpoint dict to resume from (oracle rollouts).
    rule_schedule(step, opt): may call opt.set_rule(...) before each step.
    callback(step, model, bl, opt, batch): called after backward, before the update.
    """
    torch.set_num_threads(cfg.get("threads", 2))
    data = get_data(cfg.get("data", "char"))
    model, bl, opt = build(cfg, data)
    s0 = 0
    if start is not None:
        model.load_state_dict(start["model"])
        opt.load_state_dict(start["opt"])
        s0 = start["step"]
    B, T = cfg["batch_size"], cfg["model"]["ctx"]
    dseed = cfg.get("data_seed", cfg["seed"])
    ev = data.eval_batches(cfg["eval_batches"], B, T)
    ckpts = set(cfg.get("checkpoints", []))
    hist, run_loss, n_loss, t0, diverged = [], 0.0, 0, time.time(), False
    prev = None
    for step in range(s0, cfg["steps"]):
        if rule_schedule is not None:
            rule_schedule(step, opt)
        x, y = data.batch("train", step, B, T, dseed)
        loss = model(x, y)
        if not torch.isfinite(loss) or loss.item() > 20:
            diverged = True
            log(f"diverged at step {step} (loss {loss.item()})")
            break
        for p in model.parameters():
            p.grad = None
        loss.backward()
        if callback is not None:
            callback(step, model, bl, opt, (x, y))
        if run_dir is not None and step + 1 in ckpts:
            prev = {b["name"]: (b["param"].grad.clone(), b["param"].detach().clone()) for b in bl}
        opt.step(lr_at(step, cfg))
        run_loss += loss.item()
        n_loss += 1
        if run_dir is not None and step + 1 in ckpts:
            (run_dir / "checkpoints").mkdir(exist_ok=True)
            torch.save({"step": step + 1, "model": model.state_dict(), "opt": opt.state_dict(),
                        "g_prev": {k: g for k, (g, _) in prev.items()},
                        "delta_prev": {b["name"]: b["param"].detach() - prev[b["name"]][1] for b in bl},
                        "cfg": cfg}, run_dir / "checkpoints" / f"step_{step + 1}.pt")
        if (step + 1) % cfg["eval_every"] == 0 or step + 1 == cfg["steps"]:
            val = evaluate(model, ev)
            hist.append({"step": step + 1, "train_loss": run_loss / n_loss, "val_loss": val,
                         "lr": lr_at(step, cfg), "wall_s": round(time.time() - t0, 2)})
            run_loss, n_loss = 0.0, 0
    final_val = float("inf") if diverged else (hist[-1]["val_loss"] if hist else evaluate(model, ev))
    return {"history": hist, "final_val": final_val,
            "final_train": float("inf") if diverged else (hist[-1]["train_loss"] if hist else float("nan")),
            "diverged": diverged, "wall_s": round(time.time() - t0, 2), "model": model, "opt": opt, "blocks": bl}
