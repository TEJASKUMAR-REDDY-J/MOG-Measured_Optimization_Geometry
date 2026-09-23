"""Exp 1 -- geometry atlas. Measurement only: no optimizer is changed by what is measured.

For each incumbent arm, train with instrumentation every `atlas_every` steps and log,
per 2-D block: effective dimensions of the momentum, gradient-fit theta_c/C_c for every
candidate, NS proxy error, secant curvature both the PDF's free way (consecutive
batches) and same-batch (one extra backward, isolates curvature from gradient noise),
selection scores E_c for both, half-batch gradient SNR (block and per-row), and the
input-activation participation ratio. The first arm's run also writes the
pre-declared checkpoints consumed by the oracle (Exp 2).
"""

from __future__ import annotations

import csv
import copy
from pathlib import Path

import torch

from mog.selection import atlas as A
from mog.training.trainer import train


def _grads(bl):
    return {b["name"]: b["param"].grad.clone() for b in bl}


def _set_grads(bl, grads):
    for b in bl:
        if b["name"] in grads:
            b["param"].grad = grads[b["name"]].clone()


def _grad_at(model, bl, x, y):
    for p in model.parameters():
        p.grad = None
    model(x, y).backward()
    return _grads(bl)


class Atlas:
    def __init__(self, every: int, arm: str):
        self.every, self.arm, self.cache, self.rows = every, arm, None, []

    def __call__(self, step, model, bl, opt, batch):
        if step > 0 and step % self.every == 0 and self.cache is not None:
            self.measure(step, model, bl, opt, batch)
        if (step + 1) % self.every == 0:
            self.cache = {"g": _grads(bl), "W": {b["name"]: b["param"].detach().clone() for b in bl}, "batch": batch}

    def measure(self, step, model, bl, opt, batch):
        g_now = _grads(bl)
        x, y = batch
        xp, yp = self.cache["batch"]
        g_same = _grad_at(model, bl, xp, yp)                       # g_t on batch t-1
        h = x.shape[0] // 2
        g1, g2 = _grad_at(model, bl, x[:h], y[:h]), _grad_at(model, bl, x[h:], y[h:])
        acts = {}
        hooks = [mod.register_forward_hook(lambda m, i, o, n=name: acts.__setitem__(n, i[0].detach()))
                 for name, mod in model.named_modules() if isinstance(mod, torch.nn.Linear)]
        with torch.no_grad():
            model(x)
        for hk in hooks:
            hk.remove()
        _set_grads(bl, g_now)
        for b in bl:
            name = b["name"]
            if b["param"].ndim != 2:
                continue
            B = opt.state[name]["m"]
            delta = b["param"].detach() - self.cache["W"][name]
            row = {"arm": self.arm, "step": step, "block": name, "kind": b["kind"], "layer": b["layer"],
                   "rule": opt.rules[name]}
            row.update(A.static_metrics(b, B))
            Lf = A.secants(b, g_now[name] - self.cache["g"][name], delta)
            Ls = A.secants(b, g_same[name] - self.cache["g"][name], delta)
            Ef, Es = A.scores(b, B, Lf), A.scores(b, B, Ls)
            for c in Lf:
                row.update({f"Lfree_{c}": Lf[c], f"Lsame_{c}": Ls[c], f"Efree_{c}": Ef[c], f"Esame_{c}": Es[c]})
            row["argmax_free"], row["argmax_same"] = max(Ef, key=Ef.get), max(Es, key=Es.get)
            gm, gd = (g1[name] + g2[name]) / 2, (g1[name] - g2[name])
            noise = gd.pow(2) / 4
            row["snr_block"] = float((gm.pow(2).sum() - noise.sum()).clamp_min(0) / noise.sum())
            row["snr_row_median"] = float(((gm.pow(2).sum(1) - noise.sum(1)).clamp_min(0) / noise.sum(1)).median())
            mod_name = {"head": "head"}.get(b["kind"], name.replace("L", "layers.", 1) if name.startswith("L") else None)
            row["act_pr"] = A.participation_ratio(acts[mod_name]) if mod_name in acts else float("nan")
            self.rows.append(row)


def main(cfg: dict, run_dir: Path) -> dict:
    summary = {}
    all_rows = []
    for i, (arm, rules) in enumerate(cfg["arms"].items()):
        tcfg = copy.deepcopy(cfg["train"]) | {"rules": rules, "lr": cfg["lr"][arm]}
        if i == 0:
            tcfg["checkpoints"] = cfg["checkpoints"]
        at = Atlas(cfg["atlas_every"], arm)
        r = train(tcfg, run_dir=run_dir if i == 0 else None, callback=at)
        print(f"{arm}: final val {r['final_val']:.4f} ({r['wall_s']} s)")
        all_rows += at.rows
        summary[arm] = {"final_val": r["final_val"], "final_train": r["final_train"], "diverged": r["diverged"],
                        "history": r["history"]}
    keys = list(dict.fromkeys(k for row in all_rows for k in row))
    with open(run_dir / "atlas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)
    return summary
