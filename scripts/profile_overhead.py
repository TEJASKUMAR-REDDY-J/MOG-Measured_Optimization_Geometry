"""Time one forward/backward of the 1M model and each geometry's LMO on its block shapes.

    python -m scripts.profile_overhead
"""

import torch

from mog.models.tiny_transformer import GPT, GPTConfig, blocks
from mog.optim.blockwise import ATTN, LMO, geometry_for
from mog.utils.profiling import Timer


def main():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab=65, d=128, n_layer=4, n_head=4, ctx=128))
    t = Timer()
    x = torch.randint(65, (16, 128))
    for _ in range(5):
        with t("fwd_bwd (B=16, T=128)"):
            model(x, x).backward()
    for b in blocks(model):
        if b["param"].ndim != 2:
            continue
        for rule in LMO:
            if rule == "normuon" or (rule == "spectral_head" and b["kind"] not in ATTN):
                continue
            g = geometry_for(rule, b)
            with t(f"lmo_{rule}"):
                g.lmo(b["param"].grad)
    for k, v in sorted(t.summary().items()):
        print(f"{k:24s} {v['total_s'] / v['calls'] * 1e3:8.2f} ms/call  ({v['calls']} calls)")


if __name__ == "__main__":
    main()
