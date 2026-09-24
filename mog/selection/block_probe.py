"""Per-block statistics at a checkpoint (pivot Phase 2, proposal H1). Torch models (GPT and
mog/paradigms tasks); the RL counterpart is mog/rl/probe.py.

From K independent batch gradients G_1..G_K (probe streams, never used for training) plus the
Kronecker factors of each linear/conv block (A = E[x x^T] of the block input, Gamma = E[d d^T] of
the gradient w.r.t. the block output), and the checkpoint's momentum m:

    snr            signal/noise energy per batch: mean_{i<j}<G_i,G_j> / (mean ||G_i||^2 - that)
    align_<geom>   split-batch alignment of lmo_geom(mean G_1..K/2) with mean G_K/2..K  (spectral, sign, rows)
    nuc_rank_m, stable_rank_m (/min(m,n)), elem_rank_m (/mn)   of the momentum
    nuc_rank_g     nuclear rank / min(m,n) of the K-averaged gradient
    kappa_up, kappa_lo   kappa_inf(A) bounds;  crit = kappa_up * nuclear_rank(m) / n_in (spectral beats
                   elementwise iff crit > 2/pi, proposal Sec. 4.4)
    p_out, off_out / p_in, off_in   coupling exponent of Gamma (resp. A) on m's left (right) frames
    tail           median over coordinates of the excess kurtosis of G_k (K samples)
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mog.geometry import newton_schulz
from mog.selection.block_stats import coupling_exponent, kappa_inf, ranks, split_alignment

LMOS = {"spectral": newton_schulz, "sign": torch.sign, "rows": lambda X: X / X.norm(dim=1, keepdim=True).clamp_min(1e-30)}


def _owner(model):
    return {id(m.weight): m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))}


def _factors(model, blocks, loss_fn):
    """One forward/backward with hooks -> {name: (A, Gamma)} for linear/conv blocks."""
    own, acts, outs, hooks = _owner(model), {}, {}, []
    for b in blocks:
        mod = own.get(id(b["param"]))
        if mod is None:
            continue
        name = b["name"]

        def fwd(m, inp, out, name=name):
            x = inp[0].detach()
            if isinstance(m, nn.Conv2d):
                x = F.unfold(x, m.kernel_size, padding=m.padding, stride=m.stride).transpose(1, 2)
            acts[name] = x.reshape(-1, x.shape[-1])
            out.register_hook(lambda g, name=name, conv=isinstance(m, nn.Conv2d):
                              outs.__setitem__(name, (g.transpose(1, -1) if conv else g).reshape(-1, g.shape[1] if conv else g.shape[-1])))

        hooks.append(mod.register_forward_hook(fwd))
    model.zero_grad(set_to_none=True)
    loss_fn().backward()
    for h in hooks:
        h.remove()
    return {n: (acts[n].T @ acts[n] / len(acts[n]), outs[n].T @ outs[n] / len(outs[n])) for n in acts if n in outs}


def probe(model, blocks, opt_state, loss_at, K: int = 16, kappa_iters: int = 200) -> dict:
    """loss_at(k) -> scalar loss on independent probe batch k. Returns {block: {stat: value}}."""
    G = {b["name"]: [] for b in blocks if b["param"].ndim >= 2}
    for k in range(K):
        model.zero_grad(set_to_none=True)
        loss_at(k).backward()
        for b in blocks:
            if b["name"] in G:
                G[b["name"]].append(b["param"].grad.detach().reshape(b["param"].shape[0], -1).clone())
    fac = _factors(model, blocks, lambda: loss_at(0))
    m = {n: opt_state[n]["m"].reshape(gs[0].shape) for n, gs in G.items()}
    return stats_from(G, m, fac, kappa_iters)


def stats_from(G: dict, m: dict, fac: dict, kappa_iters: int = 200) -> dict:
    """G {block: [K gradients (out, in)]}, m {block: momentum}, fac {block: (A, Gamma)}."""
    out = {}
    for name, gs in G.items():
        S = torch.stack(gs).float()
        K = len(gs)
        Gbar = S.mean(0)
        gram = torch.einsum("kij,lij->kl", S, S)
        off = (gram.sum() - gram.diagonal().sum()) / (K * (K - 1))
        noise = gram.diagonal().mean() - off
        mm = m[name].float()
        rm = ranks(mm) if mm.norm() > 0 else {"nuclear_rank": math.nan, "stable_rank": math.nan, "elementwise_rank": math.nan}
        mn = min(Gbar.shape)
        z = (S - Gbar) / S.std(0).clamp_min(1e-30)
        st = {"snr": float(off / noise.clamp_min(1e-30)),
              "nuc_rank_m": rm["nuclear_rank"] / mn, "stable_rank_m": rm["stable_rank"] / mn,
              "elem_rank_m": rm["elementwise_rank"] / Gbar.numel(), "nuc_rank_g": ranks(Gbar)["nuclear_rank"] / mn,
              "tail": float(((z ** 4).mean(0) - 3).median())}
        h = K // 2
        for g, f in LMOS.items():
            st[f"align_{g}"] = split_alignment(S[:h].mean(0), S[h:].mean(0), f)
        if name in fac and mm.norm() > 0:
            A, Gam = fac[name]
            kap = kappa_inf(A.double(), iters=kappa_iters)
            st |= {"kappa_up": kap["upper"], "kappa_lo": kap["lower"],
                   "crit": kap["upper"] * rm["nuclear_rank"] / A.shape[0]}
            po, pi = coupling_exponent(mm.double(), Gam.double()), coupling_exponent(mm.T.double(), A.double())
            st |= {"p_out": po["p"], "off_out": po["offdiag_frac"], "p_in": pi["p"], "off_in": pi["offdiag_frac"]}
        out[name] = {k: float(v) for k, v in st.items()}
    return out


def nan_to_none(d):
    return {k: ({kk: (None if isinstance(vv, float) and not np.isfinite(vv) else vv) for kk, vv in v.items()}) for k, v in d.items()}
