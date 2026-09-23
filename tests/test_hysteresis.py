from mog.selection.hysteresis import Hysteresis


def test_switch_requires_margin_and_two_consecutive_probes():
    h = Hysteresis(margin=0.1)
    assert h.update("spectral", {"spectral": 1.0, "rows": 1.05}) == "spectral"   # below margin
    assert h.update("spectral", {"spectral": 1.0, "rows": 1.2}) == "spectral"    # 1st probe
    assert h.update("spectral", {"spectral": 1.0, "rows": 1.2}) == "rows"        # 2nd probe -> switch


def test_interrupted_streak_resets():
    h = Hysteresis(margin=0.1)
    h.update("spectral", {"spectral": 1.0, "rows": 1.2})
    h.update("spectral", {"spectral": 1.0, "rows": 1.0})                         # streak broken
    assert h.update("spectral", {"spectral": 1.0, "rows": 1.2}) == "spectral"


def test_new_challenger_restarts_count():
    h = Hysteresis(margin=0.1)
    h.update("spectral", {"spectral": 1.0, "rows": 1.2, "sign": 0.5})
    assert h.update("spectral", {"spectral": 1.0, "rows": 0.5, "sign": 1.3}) == "spectral"
