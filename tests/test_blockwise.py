"""BlockOptimizer rules must match their reference definitions."""

import torch

from mog.models.tiny_transformer import GPT, GPTConfig, blocks
from mog.optim.blockwise import BlockOptimizer, resolve_rules


def _setup(spec, seed=0):
    torch.manual_seed(seed)
    model = GPT(GPTConfig(vocab=11, d=16, n_layer=1, n_head=2, ctx=8))
    bl = blocks(model)
    return model, bl, BlockOptimizer(bl, resolve_rules(bl, spec))


def _grads(model, step):
    g = torch.Generator().manual_seed(step)
    x = torch.randint(11, (4, 8), generator=g)
    for p in model.parameters():
        p.grad = None
    model(x, x).backward()


def test_resolve_priority():
    _, bl, opt = _setup({"hidden": "spectral", "q": "spectral_head", "L0.k": "rows", "default": "adam"})
    assert opt.rules["L0.q"] == "spectral_head" and opt.rules["L0.k"] == "rows"
    assert opt.rules["L0.up"] == "spectral" and opt.rules["embed"] == "adam" and opt.rules["L0.ln1"] == "adam"


def test_adam_rule_matches_torch_adam():
    m1, _, opt = _setup({"default": "adam"})
    m2, _, _ = _setup({"default": "adam"})
    ref = torch.optim.Adam(m2.parameters(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8)
    for s in range(5):
        _grads(m1, s)
        _grads(m2, s)
        opt.step(1e-2)
        ref.step()
    for a, b in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(a, b, atol=1e-6)


def test_lion_rule_matches_reference():
    m1, bl, opt = _setup({"default": "lion"})
    p = bl[2]["param"]  # L0.q
    w, m = p.detach().clone(), torch.zeros_like(p)
    for s in range(3):
        _grads(m1, s)
        g = p.grad.clone()
        w -= 1e-3 * torch.sign(0.9 * m + 0.1 * g)
        m = 0.99 * m + 0.01 * g
        opt.step(1e-3)
    assert torch.allclose(p, w, atol=1e-7)


def test_lmo_rules_are_rms_matched():
    for rule in ["sign", "euclid", "rows", "cols", "spectral", "spectral_head", "pw05", "normuon"]:
        model, bl, opt = _setup({"attn": rule, "default": "adam"})
        _grads(model, 0)
        b = bl[2]
        d = opt.direction(b, b["param"].grad, opt.state[b["name"]] | {"t": 1}, rule)
        assert abs(float(d.pow(2).mean().sqrt()) - 0.2) < 1e-4, rule


def test_adam_mini_is_blockwise_constant():
    model, bl, opt = _setup({"default": "adam_mini"})
    _grads(model, 0)
    opt.step(1e-2)
    q = next(b for b in bl if b["name"] == "L0.q")
    st = opt.state["L0.q"]
    # one effective v per head: m / d is constant within each head block
    from mog.optim.blockwise import _adam_mini_mean
    vm = _adam_mini_mean(st["v"], "q", 2).reshape(2, -1)
    assert torch.allclose(vm, vm[:, :1].expand_as(vm))


def test_svd_rules_keep_singular_vectors_and_map_spectrum():
    from mog.optim.blockwise import svd_rule

    u = torch.randn(12, 7, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    U, S, Vh = torch.linalg.svd(u, full_matrices=False)
    polar = svd_rule(u, "polar_svd")
    assert torch.allclose(polar, U @ Vh)
    for rule, expect in [("freon34", S ** -0.5), ("freon23", S ** (-1 / 3))]:
        f = torch.diagonal(U.T @ svd_rule(u, rule) @ Vh.T)
        assert torch.allclose(f, expect, rtol=1e-6)
    x = S / u.norm()
    for _ in range(5):
        x = 4.1 * x * (1 - x * x) ** 2
    assert torch.allclose(torch.diagonal(U.T @ svd_rule(u, "kaon") @ Vh.T), x)
    r = torch.diagonal(U.T @ svd_rule(u, "randspec", torch.Generator().manual_seed(1)) @ Vh.T)
    assert ((r >= 0) & (r <= 1)).all() and r.std() > 0


def test_scale_controls():
    """adam_rms = Adam direction at RMS 0.2; randspec is deterministic per (block, step)."""
    for rule in ("adam_rms", "randspec", "kaon", "freon23", "polar_svd"):
        m, bl, opt = _setup({"hidden": rule, "default": "adam"})
        _grads(m, 0)
        b = next(b for b in bl if b["kind"] == "up")
        st = {"m": torch.zeros_like(b["param"]), "v": b["param"].grad ** 2 * 0.05, "t": 1}
        d = opt.direction(b, b["param"].grad, st, rule)
        assert abs(d.pow(2).mean().sqrt().item() - 0.2) < 1e-5
        if rule == "adam_rms":
            a = opt.direction(b, b["param"].grad, {"m": torch.zeros_like(d), "v": st["v"], "t": 1}, "adam")
            assert torch.allclose(d / d.norm(), a / a.norm(), atol=1e-6)
    m, bl, opt = _setup({"hidden": "randspec", "default": "adam"})
    _grads(m, 0)
    b = bl[2]
    mk = lambda: {"m": torch.zeros_like(b["param"]), "v": torch.zeros_like(b["param"]), "t": 3}
    assert torch.equal(opt.direction(b, b["param"].grad, mk(), "randspec"), opt.direction(b, b["param"].grad, mk(), "randspec"))
