"""Index every run under results/raw into results/processed/runs_index.csv.

    python -m scripts.aggregate_results
"""

from __future__ import annotations

import csv
import json

from mog.utils.reproducibility import REPO_ROOT

FIELDS = ["experiment_id", "run", "status", "evidence_class", "seed", "git_commit", "git_dirty", "wall_time_s", "command"]


def main() -> None:
    raw, out = REPO_ROOT / "results" / "raw", REPO_ROOT / "results" / "processed" / "runs_index.csv"
    rows = []
    for f in sorted(raw.glob("*/*/results.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        rows.append({k: r.get(k) for k in FIELDS} | {"run": f.parent.name})
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"indexed {len(rows)} runs -> {out}")


if __name__ == "__main__":
    main()
