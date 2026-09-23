"""Build a model from a config dict: {"type": "tiny_transformer" | "tiny_mlp", ...}."""

from __future__ import annotations

from .tiny_mlp import TinyMLP
from .tiny_transformer import GPT, GPTConfig


def build_model(cfg: dict, vocab: int | None = None):
    kind = cfg.get("type", "tiny_transformer")
    args = {k: v for k, v in cfg.items() if k != "type"}
    if kind == "tiny_transformer":
        return GPT(GPTConfig(vocab=vocab, **args))
    if kind == "tiny_mlp":
        return TinyMLP(**args)
    raise ValueError(f"unknown model type {kind!r}")
