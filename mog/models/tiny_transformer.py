"""Small pre-LN GPT with separately named q/k/v/o/up/down projections.

Every 2-D weight is its own "block" with a `kind` tag, which is what the
per-block optimizer, the atlas and the oracle key on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab: int
    d: int = 128
    n_layer: int = 4
    n_head: int = 4
    ctx: int = 128
    mlp_mult: int = 4


class Layer(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.n_head = c.n_head
        self.ln1 = nn.LayerNorm(c.d, bias=False)
        self.q, self.k, self.v, self.o = (nn.Linear(c.d, c.d, bias=False) for _ in range(4))
        self.ln2 = nn.LayerNorm(c.d, bias=False)
        self.up = nn.Linear(c.d, c.mlp_mult * c.d, bias=False)
        self.down = nn.Linear(c.mlp_mult * c.d, c.d, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        h = self.ln1(x)
        q, k, v = (f(h).view(B, T, self.n_head, -1).transpose(1, 2) for f in (self.q, self.k, self.v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(y.transpose(1, 2).reshape(B, T, D))
        return x + self.down(F.gelu(self.up(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.cfg = c
        self.embed = nn.Embedding(c.vocab, c.d)
        self.pos = nn.Embedding(c.ctx, c.d)
        self.layers = nn.ModuleList(Layer(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.d, bias=False)
        self.head = nn.Linear(c.d, c.vocab, bias=False)
        for name, p in self.named_parameters():
            if p.ndim == 2:
                std = 0.02 / math.sqrt(2 * c.n_layer) if name.endswith(("o.weight", "down.weight")) else 0.02
                nn.init.normal_(p, std=std)

    def forward(self, idx, targets=None):
        T = idx.size(1)
        x = self.embed(idx) + self.pos(torch.arange(T, device=idx.device))
        for layer in self.layers:
            x = layer(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


def blocks(model: GPT) -> list[dict]:
    """Named parameter blocks: {'name', 'param', 'kind', 'layer', 'n_head'}."""
    out = [
        {"name": "embed", "param": model.embed.weight, "kind": "embed", "layer": -1},
        {"name": "pos", "param": model.pos.weight, "kind": "pos", "layer": -1},
    ]
    for i, L in enumerate(model.layers):
        for kind in ("q", "k", "v", "o", "up", "down"):
            out.append({"name": f"L{i}.{kind}", "param": getattr(L, kind).weight, "kind": kind, "layer": i})
        for g in ("ln1", "ln2"):
            out.append({"name": f"L{i}.{g}", "param": getattr(L, g).weight, "kind": "gain", "layer": i})
    out.append({"name": "ln_f", "param": model.ln_f.weight, "kind": "gain", "layer": model.cfg.n_layer})
    out.append({"name": "head", "param": model.head.weight, "kind": "head", "layer": model.cfg.n_layer})
    for b in out:
        b["n_head"] = model.cfg.n_head
    return out
