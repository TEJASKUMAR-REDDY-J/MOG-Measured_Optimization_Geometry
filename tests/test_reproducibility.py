"""Runner produces the full artifact set, never overwrites, and is deterministic given the seed."""

import json

import pytest

from mog.utils.reproducibility import REPO_ROOT, new_run_dir
from scripts.run_experiment import run

SMOKE = REPO_ROOT / "configs" / "synthetic" / "exp000_smoke.yaml"


def test_run_dir_never_overwrites(tmp_path):
    dirs = [new_run_dir(tmp_path, "x") for _ in range(20)]  # back-to-back, same clock tick
    assert len(set(dirs)) == 20
    a = dirs[0]
    with pytest.raises(FileExistsError):
        a.mkdir(parents=True, exist_ok=False)


def test_smoke_run_artifacts_and_determinism(tmp_path):
    r1 = run(SMOKE, tmp_path)
    r2 = run(SMOKE, tmp_path)
    for name in ["config.yaml", "environment.json", "stdout.log", "results.json", "README.md", "metrics.csv"]:
        assert (r1 / name).is_file(), name
    assert any((r1 / "figures").glob("*.png"))
    res = json.loads((r1 / "results.json").read_text())
    assert res["status"] == "ok" and res["evidence_class"] == "smoke"
    assert (r1 / "metrics.csv").read_text() == (r2 / "metrics.csv").read_text()
