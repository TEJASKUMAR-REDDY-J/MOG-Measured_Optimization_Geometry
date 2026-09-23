"""Exp 2 -- oracle geometry per block at frozen checkpoints, and proxy validation (H1, H2).

Oracle: from checkpoint W_s (incumbent = folklore Muon hybrid), switch ONE block to
candidate rule c, keep every other block on its incumbent rule, train h steps on a
fixed data stream at the checkpoint's LR (held constant), and measure the loss
reduction on the fixed validation batches. Same data for every candidate -> paired.
Robustness: incumbent and best candidate are re-run on a second data stream.

Proxies scored against the oracle (per block, across candidates):
  free      E_c = theta_c / L_hat_c, secant from (g_{s-1}, g_s, Delta_{s-1}) -- the PDF's proxy
  same      same, but g_s evaluated on batch s-1 (isolates curvature from gradient noise)
  own       secant along the candidate's OWN step on one batch (one extra backward)
  fit       theta_c / C_c alone (gradient-fit, no curvature)
  lookahead realized one-step loss change on the current batch (one extra forward)
adam/adam_mini are proxied by the sign geometry and normuon by spectral (PDF §5 convention);
lookahead uses each rule's real direction.
"""

from __future__ import annotations

import copy
import csv
import math
from pathlib import Path

import numpy as np
import torch

from mog.selection import atlas as A
from mog.training.trainer import build, evaluate, get_data, lr_at, train
from mog.utils.reproducibility import REPO_ROOT

CANDIDATES = {
    "attn": ["adam", "sign", "euclid", "rows", "cols", "spectral", "spectral_head", "normuon", "pw05"],
    "mlp": ["adam", "sign", "euclid", "rows", "cols", "spectral", "normuon", "pw05"],
    "vocab": ["adam", "sign", "euclid", "rows", "cols", "spectral"],
}
GROUP = {"q": "attn", "k": "attn", "v": "attn", "o": "attn", "up": "mlp", "down": "mlp",
         "embed": "vocab", "pos": "vocab", "head": "vocab"}


def proxy_geom(rule):
    return A.PROXY_GEOMETRY.get(rule, rule)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def rollout(ck, cfg, block, rule, data_seed, ev):
    rcfg = copy.deepcopy(ck["cfg"])
    rcfg.update(steps=ck["step"] + cfg["horizon"], data_seed=data_seed, eval_every=10**9,
                const_lr=lr_at(ck["step"] - 1, ck["cfg"]), checkpoints=[])
    sched = (lambda s, opt: opt.set_rule(block, rule)) if rule is not None else None
    r = train(rcfg, start=ck, rule_schedule=sched, log=lambda *a: None)
    return float("inf") if r["diverged"] else evaluate(r["model"], ev)


def proxies(ck, cfg, ev_unused):
    """Per (block, candidate) proxy scores at the checkpoint."""
    data = get_data()
    ccfg = ck["cfg"]
    model, bl, opt = build(ccfg, data)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    B_, T = ccfg["batch_size"], ccfg["model"]["ctx"]
    s = ck["step"]
    x, y = data.batch("train", s, B_, T, ccfg.get("data_seed", ccfg["seed"]))
    xp, yp = data.batch("train", s - 1, B_, T, ccfg.get("data_seed", ccfg["seed"]))

    def grads(xx, yy):
        for p in model.parameters():
            p.grad = None
        loss = model(xx, yy)
        loss.backward()
        return float(loss), {b["name"]: b["param"].grad.clone() for b in bl}

    loss0, g_now = grads(x, y)
    _, g_same = grads(xp, yp)
    lr = lr_at(s - 1, ccfg)
    out = {}
    for b in bl:
        if b["param"].ndim != 2:
            continue
        name, W0 = b["name"], b["param"].detach().clone()
        Bm = opt.state[name]["m"]
        delta = ck["delta_prev"][name]
        Lf = A.secants(b, g_now[name] - ck["g_prev"][name], delta)
        Ls = A.secants(b, g_same[name] - ck["g_prev"][name], delta)
        Ef, Es = A.scores(b, Bm, Lf), A.scores(b, Bm, Ls)
        for rule in CANDIDATES[GROUP[b["kind"]]]:
            pg = proxy_geom(rule)
            geo = A.candidate_geometry(pg, b)
            st = copy.deepcopy(opt.state[name])
            st["t"] += 1
            st["v"].mul_(0.95).addcmul_(g_now[name], g_now[name], value=0.05)
            step_vec = -lr * opt.direction(b, g_now[name], st, rule)
            with torch.no_grad():
                b["param"].add_(step_vec)
            l1, g1 = grads(x, y)
            with torch.no_grad():
                b["param"].copy_(W0)
            dual = geo.exact_dual_norm(g1[name] - g_now[name]) if hasattr(geo, "exact_dual_norm") else geo.dual_norm(g1[name] - g_now[name])
            L_own = float(dual) / float(geo.norm(step_vec))
            theta = float((geo.exact_dual_norm(Bm) if hasattr(geo, "exact_dual_norm") else geo.dual_norm(Bm)) ** 2 / Bm.norm() ** 2)
            out[(name, rule)] = {"free": Ef[pg], "same": Es[pg], "own": theta / L_own,
                                 "fit": theta / geo.C(Bm.shape), "lookahead": loss0 - l1}
    return out


def main(cfg: dict, run_dir: Path) -> dict:
    src = cfg["source_run"]
    if src.startswith("latest:"):
        src = sorted((REPO_ROOT / "results" / "raw" / src.split(":", 1)[1]).iterdir())[-1]
    src = REPO_ROOT / src
    print(f"source run: {src}")
    data = get_data()
    rows = []
    for s in cfg["checkpoints"]:
        ck = torch.load(src / "checkpoints" / f"step_{s}.pt", weights_only=False)
        ccfg = ck["cfg"]
        ev = data.eval_batches(ccfg["eval_batches"], ccfg["batch_size"], ccfg["model"]["ctx"])
        model, bl, opt = build(ccfg, data)
        model.load_state_dict(ck["model"])
        L0 = evaluate(model, ev)
        base = rollout(ck, cfg, None, None, cfg["data_seed"], ev)            # everyone incumbent
        base_alt = rollout(ck, cfg, None, None, cfg["data_seed_alt"], ev)
        px = proxies(ck, cfg, ev)
        print(f"ckpt {s}: L0={L0:.4f} incumbent-rollout={base:.4f}")
        for b in bl:
            if b["param"].ndim != 2:
                continue
            name, inc = b["name"], ck["opt"]["rules"][b["name"]]
            res = {}
            for rule in CANDIDATES[GROUP[b["kind"]]]:
                res[rule] = base if rule == inc else rollout(ck, cfg, name, rule, cfg["data_seed"], ev)
            best = min(res, key=res.get)
            best_alt = rollout(ck, cfg, name, best, cfg["data_seed_alt"], ev) if best != inc else base_alt
            for rule, Lh in res.items():
                p = px[(name, rule)]
                rows.append({"ckpt": s, "block": name, "kind": b["kind"], "layer": b["layer"], "incumbent": inc,
                             "rule": rule, "L0": L0, "L_after": Lh, "gain_vs_incumbent": base - Lh,
                             "is_best": rule == best,
                             "best_gain_alt": (base_alt - best_alt) if rule == best else "",
                             **{f"proxy_{k}": v for k, v in p.items()}})
            print(f"  {name:8s} inc={inc:9s} best={best:13s} gain={base - res[best]:+.5f} alt={base_alt - best_alt:+.5f}")
    keys = list(rows[0])
    with open(run_dir / "oracle.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return analyze(rows) | {"source_run": str(src.relative_to(REPO_ROOT))}


def analyze(rows) -> dict:
    groups = {}
    for r in rows:
        groups.setdefault((r["ckpt"], r["block"]), []).append(r)
    per_proxy = {}
    for key in ("free", "same", "own", "fit", "lookahead"):
        rhos, top1, regret, chance = [], [], [], []
        for (_, _), rs in groups.items():
            gains = np.array([r["gain_vs_incumbent"] for r in rs])
            sc = np.array([r[f"proxy_{key}"] for r in rs], dtype=float)
            if not np.all(np.isfinite(sc)):
                continue
            rhos.append(spearman(sc, gains))
            pick = int(np.argmax(sc))
            top1.append(pick == int(np.argmax(gains)))
            chance.append(1 / len(rs))
            best = gains.max()
            regret.append(best - gains[pick])
        rhos = [x for x in rhos if not math.isnan(x)]
        per_proxy[key] = {"spearman_median": float(np.median(rhos)), "spearman_iqr": [float(np.percentile(rhos, 25)), float(np.percentile(rhos, 75))],
                          "top1_agreement": float(np.mean(top1)), "top1_chance": float(np.mean(chance)),
                          "mean_regret": float(np.mean(regret)), "n_blocks": len(rhos)}
    bests = [r for r in rows if r["is_best"]]
    changed = [r for r in bests if r["rule"] != r["incumbent"]]
    stable = [r for r in changed if r["best_gain_alt"] != "" and float(r["best_gain_alt"]) > 0]
    return {"proxies": per_proxy, "n_block_ckpts": len(bests),
            "oracle_prefers_non_incumbent": len(changed), "non_incumbent_stable_on_alt_data": len(stable),
            "best_rule_counts": {k: sum(r["rule"] == k for r in bests) for k in sorted({r["rule"] for r in bests})},
            "median_best_gain": float(np.median([r["gain_vs_incumbent"] for r in bests]))}
