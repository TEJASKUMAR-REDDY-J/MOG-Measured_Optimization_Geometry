"""Per-block optimizer: every parameter block carries its own update rule, and a
block's rule can be switched at any step (used by the oracle and by MOG arms).

Rules
-----
native (their usual definitions, own scale):
    sgd        Nesterov EMA momentum SGD (beta 0.9)
    adam       Adam (beta1 0.9, beta2 0.95, bias-corrected, eps 1e-8), no weight decay
    adam_mini  Adam with v averaged over Adam-mini blocks (per head for q/k,
               per output neuron for v/o/mlp, per token for embed/pos/head)
    lion       Lion (beta1 0.9 interpolation, beta2 0.99 EMA)
geometry / LMO (Nesterov momentum beta 0.95 -> Geometry.lmo -> rescaled to RMS 0.2,
the Moonlight/NorMuon convention that lets them share one LR with Adam):
    sign, euclid, rows, cols, spectral (Muon, Newton-Schulz), spectral_head
    (per-head grouped; q/k/v by output rows, o by input cols), pw05 (alpha=1/2
    partial whitening, exact SVD), normuon (NS + per-row second moment, beta2 0.95)

Every block also keeps an Adam second moment v, so switching to adam is seamless.
Weight decay is 0 for all rules (one less tuned knob; documented in the protocol).
"""

from __future__ import annotations

import math
import zlib

import torch

from mog.geometry import Elementwise, Euclidean, RowNorm, Spectral, newton_schulz

NATIVE = ("sgd", "adam", "adam_mini", "lion")
LMO = ("sign", "euclid", "rows", "cols", "spectral", "spectral_head", "pw05", "normuon", "spectral_randgroup")
RULES = NATIVE + LMO
MATRIX_KINDS = ("embed", "pos", "q", "k", "v", "o", "up", "down", "head")
HIDDEN = ("q", "k", "v", "o", "up", "down")
ATTN = ("q", "k", "v", "o")


def geometry_for(rule: str, block: dict):
    kind, H, shape = block["kind"], block["n_head"], block["param"].shape
    if rule == "sign":
        return Elementwise()
    if rule == "euclid":
        return Euclidean()
    if rule == "rows":
        return RowNorm("rows")
    if rule == "cols":
        return RowNorm("cols")
    if rule == "spectral":
        return Spectral()
    if rule == "pw05":
        return Spectral(alpha=0.5, method="svd")
    if rule == "spectral_head":
        if kind not in ATTN:
            raise ValueError("spectral_head only applies to attention projections")
        return Spectral(k=shape[1] // H, axis="cols") if kind == "o" else Spectral(k=shape[0] // H)
    if rule == "spectral_randgroup":  # H4 control: head-sized groups, random membership (fixed per block)
        if kind not in ATTN:
            raise ValueError("spectral_randgroup only applies to attention projections")
        from mog.geometry.grouped import PermutedGroupedSpectral

        axis, n = ("cols", shape[1]) if kind == "o" else ("rows", shape[0])
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(zlib.crc32(block["name"].encode())))
        return PermutedGroupedSpectral(k=n // H, perm=perm, axis=axis)
    raise ValueError(rule)


def resolve_rules(blocks: list[dict], spec: dict) -> dict[str, str]:
    """Map block name -> rule. Priority: name > kind > attn/mlp > hidden > matrix > default."""
    out = {}
    for b in blocks:
        name, kind = b["name"], b["kind"]
        keys = [name, kind]
        if kind in ATTN:
            keys.append("attn")
        if kind in ("up", "down"):
            keys.append("mlp")
        if kind in HIDDEN:
            keys.append("hidden")
        if kind in MATRIX_KINDS:
            keys.append("matrix")
        keys.append("default")
        rule = next(spec[k] for k in keys if k in spec)
        if kind == "gain" and rule in LMO:
            raise ValueError(f"{name}: LMO rule {rule} is not defined for 1-D gains")
        out[name] = rule
    return out


def _adam_mini_mean(v: torch.Tensor, kind: str, n_head: int) -> torch.Tensor:
    if v.ndim == 1:
        return v.mean().expand_as(v)
    if kind in ("q", "k"):
        r = v.reshape(n_head, -1)
        return r.mean(1, keepdim=True).expand_as(r).reshape(v.shape)
    return v.mean(1, keepdim=True).expand_as(v)


class BlockOptimizer:
    def __init__(self, blocks: list[dict], rules: dict[str, str], rms: float = 0.2):
        self.blocks = blocks
        self.rules = dict(rules)
        self.rms = rms
        self.state = {b["name"]: {"m": torch.zeros_like(b["param"]), "v": torch.zeros_like(b["param"]), "t": 0}
                      for b in blocks}
        self._geo = {}

    def set_rule(self, name: str, rule: str) -> None:
        self.rules[name] = rule

    def _geometry(self, rule, block):
        key = (rule, block["name"])
        if key not in self._geo:
            self._geo[key] = geometry_for(rule, block)
        return self._geo[key]

    def state_dict(self) -> dict:
        return {"rules": dict(self.rules),
                "state": {k: {kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in s.items()}
                          for k, s in self.state.items()}}

    def load_state_dict(self, sd: dict) -> None:
        self.rules = dict(sd["rules"])
        for k, s in sd["state"].items():
            self.state[k] = {kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in s.items()}

    def direction(self, block: dict, g: torch.Tensor, st: dict, rule: str) -> torch.Tensor:
        """Compute the update direction and advance this block's momentum state."""
        m, v, t = st["m"], st["v"], st["t"]
        if rule in ("adam", "adam_mini"):
            m.lerp_(g, 0.1)
            vv = v if rule == "adam" else _adam_mini_mean(v, block["kind"], block["n_head"])
            return (m / (1 - 0.9**t)) / ((vv / (1 - 0.95**t)).sqrt() + 1e-8)
        if rule == "lion":
            c = m * 0.9 + g * 0.1
            m.lerp_(g, 0.01)
            return torch.sign(c)
        if rule == "sgd":
            m.lerp_(g, 0.1)
            return g.lerp(m, 0.9)
        m.lerp_(g, 0.05)
        u = g.lerp(m, 0.95)
        if rule == "normuon":
            O = newton_schulz(u)
            vr = st.setdefault("vrow", torch.zeros(O.shape[0], 1))
            vr.lerp_((O * O).mean(1, keepdim=True), 0.05)
            O = O / ((vr / (1 - 0.95**t)).sqrt() + 1e-8)
        else:
            O = self._geometry(rule, block).lmo(u)
        return O * (self.rms * math.sqrt(O.numel()) / O.norm().clamp_min(1e-30))

    @torch.no_grad()
    def step(self, lr: float) -> None:
        for b in self.blocks:
            p = b["param"]
            if p.grad is None:
                continue
            st = self.state[b["name"]]
            st["t"] += 1
            st["v"].mul_(0.95).addcmul_(p.grad, p.grad, value=0.05)
            d = self.direction(b, p.grad, st, self.rules[b["name"]])
            p.add_(d, alpha=-lr)
