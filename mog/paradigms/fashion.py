"""FashionMNIST tasks for three learning paradigms (pivot Phase 2).

    supervised   small CNN classifier, cross-entropy
    ssl          SimCLR (NT-Xent) on the same CNN trunk + projector; eval also reports a
                 ridge linear probe and RankMe
    flow         rectified-flow velocity MLP (generative); eval also reports loss per t-bin

Every task: build(seed) -> (model, blocks); loss(model, step, data_seed); evaluate(model) ->
{"val": primary held-out loss, ...}. Batches depend only on (data_seed, step), so every oracle
candidate sees identical data. Data: the four official gz files in data/fashion_mnist/ (SHA-256 below).
"""

from __future__ import annotations

import gzip
import hashlib
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mog.utils.reproducibility import REPO_ROOT

DATA_DIR = REPO_ROOT / "data" / "fashion_mnist"
SHA256 = {
    "train-images-idx3-ubyte.gz": "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
    "train-labels-idx1-ubyte.gz": "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
    "t10k-images-idx3-ubyte.gz": "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
    "t10k-labels-idx1-ubyte.gz": "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
}
EVAL_SEED = 1_000_000
_DATA = {}


def _read(name, offset):
    raw = (DATA_DIR / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA256[name]:
        raise ValueError(f"{name}: SHA-256 mismatch")
    return np.frombuffer(gzip.decompress(raw), np.uint8, offset=offset)


def fashion():
    if not _DATA:
        for split, pre in (("train", "train"), ("test", "t10k")):
            x = torch.tensor(_read(f"{pre}-images-idx3-ubyte.gz", 16).reshape(-1, 1, 28, 28), dtype=torch.float32) / 255
            _DATA[split] = (x * 2 - 1, torch.tensor(_read(f"{pre}-labels-idx1-ubyte.gz", 8), dtype=torch.long))
    return _DATA


def _gen(data_seed, step):
    return torch.Generator().manual_seed(data_seed * 1_000_003 + step)


def _blk(name, param, kind, role, layer):
    return {"name": name, "param": param, "kind": kind, "role": role, "layer": layer, "n_head": 1}


def _gain_blocks(model, taken):
    return [_blk(n, p, "gain", "gain", -1) for n, p in model.named_parameters() if n not in taken]


class Trunk(nn.Module):
    """conv(1->c1) -> conv(c1->c2, s2) -> conv(c2->c2, s2) -> fc(c2*49 -> d)."""

    def __init__(self, c1=16, c2=32, d=128):  # ponytail: CPU scale; widen with compute
        super().__init__()
        self.conv1 = nn.Conv2d(1, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(c2, c2, 3, stride=2, padding=1)
        self.fc = nn.Linear(c2 * 49, d)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return F.relu(self.fc(x.flatten(1)))


def _trunk_blocks(prefix, t):
    return [_blk(f"{prefix}conv1.weight", t.conv1.weight, "conv_in", "boundary", 0),
            _blk(f"{prefix}conv2.weight", t.conv2.weight, "conv", "hidden", 1),
            _blk(f"{prefix}conv3.weight", t.conv3.weight, "conv", "hidden", 2),
            _blk(f"{prefix}fc.weight", t.fc.weight, "fc", "hidden", 3)]


class Supervised:
    def __init__(self, cfg):
        self.B = cfg.get("batch_size", 128)
        self.n_eval = cfg.get("eval_size", 2000)

    def build(self, seed):
        torch.manual_seed(seed)
        model = nn.ModuleDict({"trunk": Trunk(), "head": nn.Linear(128, 10)})
        bl = _trunk_blocks("trunk.", model["trunk"]) + [_blk("head.weight", model["head"].weight, "head", "boundary", 4)]
        return model, bl + _gain_blocks(model, {b["name"] for b in bl})

    def loss(self, model, step, data_seed):
        x, y = fashion()["train"]
        i = torch.randint(len(x), (self.B,), generator=_gen(data_seed, step))
        return F.cross_entropy(model["head"](model["trunk"](x[i])), y[i])

    @torch.no_grad()
    def evaluate(self, model):
        x, y = fashion()["test"]
        logits = model["head"](model["trunk"](x[: self.n_eval]))
        return {"val": float(F.cross_entropy(logits, y[: self.n_eval])),
                "acc": float((logits.argmax(1) == y[: self.n_eval]).float().mean())}


def augment(x, g):
    """Random shift (pad 3, crop 28), horizontal flip, additive noise; deterministic given g."""
    n = len(x)
    xp = F.pad(x, (3, 3, 3, 3), value=-1.0)
    dx, dy = torch.randint(0, 7, (n,), generator=g), torch.randint(0, 7, (n,), generator=g)
    idx = torch.arange(28)
    rows = (dy[:, None] + idx)[:, :, None].expand(n, 28, 28)
    cols = (dx[:, None] + idx)[:, None, :].expand(n, 28, 28)
    out = xp[torch.arange(n)[:, None, None], 0, rows, cols][:, None]
    flip = torch.rand(n, generator=g) < 0.5
    out[flip] = out[flip].flip(-1)
    return out + 0.1 * torch.randn(out.shape, generator=g)


def nt_xent(z1, z2, tau=0.2):
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    n = len(z1)
    sim = z @ z.T / tau
    sim.fill_diagonal_(float("-inf"))
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(n)])
    return F.cross_entropy(sim, target)


class SSL:
    def __init__(self, cfg):
        self.B = cfg.get("batch_size", 128)
        self.n_eval = cfg.get("eval_size", 1024)
        self.n_probe = cfg.get("probe_size", 5000)

    def build(self, seed):
        torch.manual_seed(seed)
        model = nn.ModuleDict({"trunk": Trunk(), "proj1": nn.Linear(128, 128), "proj2": nn.Linear(128, 64)})
        bl = _trunk_blocks("trunk.", model["trunk"])
        bl += [_blk("proj1.weight", model["proj1"].weight, "proj", "hidden", 4),
               _blk("proj2.weight", model["proj2"].weight, "proj_out", "boundary", 5)]
        return model, bl + _gain_blocks(model, {b["name"] for b in bl})

    def _z(self, model, x):
        return model["proj2"](F.relu(model["proj1"](model["trunk"](x))))

    def _nce(self, model, x, g):
        return nt_xent(self._z(model, augment(x, g)), self._z(model, augment(x, g)))

    def loss(self, model, step, data_seed):
        x, _ = fashion()["train"]
        g = _gen(data_seed, step)
        return self._nce(model, x[torch.randint(len(x), (self.B,), generator=g)], g)

    @torch.no_grad()
    def evaluate(self, model, probe=False):
        xt, yt = fashion()["test"]
        val = float(self._nce(model, xt[: self.n_eval], _gen(EVAL_SEED, 0)))
        out = {"val": val}
        if probe:
            xtr, ytr = fashion()["train"]
            ftr, fte = model["trunk"](xtr[: self.n_probe]), model["trunk"](xt)
            s = torch.linalg.svdvals(fte - fte.mean(0))
            p = s / s.sum()
            out["rankme"] = float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()))
            mu, sd = ftr.mean(0), ftr.std(0) + 1e-6
            A = torch.cat([(ftr - mu) / sd, torch.ones(len(ftr), 1)], 1)
            W = torch.linalg.solve(A.T @ A + 1e-2 * len(A) * torch.eye(A.shape[1]), A.T @ F.one_hot(ytr[: self.n_probe], 10).float())
            pred = (torch.cat([(fte - mu) / sd, torch.ones(len(fte), 1)], 1) @ W).argmax(1)
            out["probe_acc"] = float((pred == yt).float().mean())
        return out


def time_features(t, dim=32):
    f = torch.exp(torch.linspace(0, math.log(1000), dim // 2))
    return torch.cat([torch.sin(t[:, None] * f), torch.cos(t[:, None] * f)], 1)


class VelocityMLP(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.inp = nn.Linear(784 + 32, width)
        self.h1 = nn.Linear(width, width)
        self.h2 = nn.Linear(width, width)
        self.out = nn.Linear(width, 784)

    def forward(self, x, t):
        h = F.silu(self.inp(torch.cat([x.flatten(1), time_features(t)], 1)))
        h = F.silu(self.h1(h))
        h = F.silu(self.h2(h))
        return self.out(h).view_as(x)


class Flow:
    N_BINS = 8

    def __init__(self, cfg):
        self.B = cfg.get("batch_size", 128)
        self.n_eval = cfg.get("eval_size", 2000)

    def build(self, seed):
        torch.manual_seed(seed)
        model = VelocityMLP()
        bl = [_blk("inp.weight", model.inp.weight, "fm_in", "boundary", 0),
              _blk("h1.weight", model.h1.weight, "fm_hidden", "hidden", 1),
              _blk("h2.weight", model.h2.weight, "fm_hidden", "hidden", 2),
              _blk("out.weight", model.out.weight, "fm_out", "boundary", 3)]
        return model, bl + _gain_blocks(model, {b["name"] for b in bl})

    @staticmethod
    def _fm(model, x1, g, t=None):
        x0 = torch.randn(x1.shape, generator=g)
        t = torch.rand(len(x1), generator=g) if t is None else t
        xt = (1 - t)[:, None, None, None] * x0 + t[:, None, None, None] * x1
        return ((model(xt, t) - (x1 - x0)) ** 2).flatten(1).mean(1)

    def loss(self, model, step, data_seed):
        x, _ = fashion()["train"]
        g = _gen(data_seed, step)
        return self._fm(model, x[torch.randint(len(x), (self.B,), generator=g)], g).mean()

    @torch.no_grad()
    def evaluate(self, model):
        x = fashion()["test"][0][: self.n_eval]
        t = (torch.arange(self.n_eval) % self.N_BINS + 0.5) / self.N_BINS  # stratified t, one value per bin
        per = self._fm(model, x, _gen(EVAL_SEED, 0), t)
        bins = torch.arange(self.n_eval) % self.N_BINS
        return {"val": float(per.mean())} | {f"t{k}": float(per[bins == k].mean()) for k in range(self.N_BINS)}


TASKS = {"supervised": Supervised, "ssl": SSL, "flow": Flow}
