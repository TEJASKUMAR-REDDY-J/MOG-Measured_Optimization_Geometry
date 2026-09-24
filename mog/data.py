"""Character-level Tiny Shakespeare with a stateless, seekable batch stream.

Batch t of stream `data_seed` depends only on (data_seed, t), so a run can be
resumed from any checkpoint and every oracle candidate can see the identical
data sequence.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from mog.utils.reproducibility import REPO_ROOT

DATA_PATH = REPO_ROOT / "data" / "tinyshakespeare.txt"
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
EVAL_STREAM = 1_000_000  # fixed data_seed for the shared validation batches


class CharData:
    def __init__(self, path: Path = DATA_PATH, val_frac: float = 0.1):
        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
            raise ValueError(f"{path} does not match the recorded SHA-256; re-download from {DATA_URL}")
        text = raw.decode("utf-8")
        self.chars = sorted(set(text))
        self.vocab = len(self.chars)
        data = torch.tensor(list(raw), dtype=torch.long)  # text is ASCII: byte == char
        lut = torch.full((256,), -1, dtype=torch.long)
        lut[torch.tensor([ord(c) for c in self.chars])] = torch.arange(self.vocab)
        data = lut[data]
        n = int(len(data) * (1 - val_frac))
        self.split = {"train": data[:n], "val": data[n:]}
        self.token_counts = torch.bincount(self.split["train"], minlength=self.vocab)

    def batch(self, split: str, step: int, B: int, T: int, data_seed: int):
        d = self.split[split]
        g = torch.Generator().manual_seed(data_seed * 1_000_003 + step)
        ix = torch.randint(len(d) - T - 1, (B,), generator=g)
        x = torch.stack([d[i:i + T] for i in ix])
        y = torch.stack([d[i + 1:i + T + 1] for i in ix])
        return x, y

    def eval_batches(self, n: int, B: int, T: int):
        return [self.batch("val", i, B, T, EVAL_STREAM) for i in range(n)]


class BPEData(CharData):
    """Same text, GPT-2 BPE (tiktoken) remapped to the dense set of token ids that occur in
    it (~11.7k types, heavy-tailed). Used for H5, which needs a realistic rare-token regime."""

    def __init__(self, path: Path = DATA_PATH, val_frac: float = 0.1):
        import tiktoken

        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
            raise ValueError(f"{path} does not match the recorded SHA-256; re-download from {DATA_URL}")
        ids = torch.tensor(tiktoken.get_encoding("gpt2").encode(raw.decode("utf-8")), dtype=torch.long)
        types, data = torch.unique(ids, return_inverse=True)
        self.vocab = len(types)
        n = int(len(data) * (1 - val_frac))
        self.split = {"train": data[:n], "val": data[n:]}
        self.token_counts = torch.bincount(self.split["train"], minlength=self.vocab)
