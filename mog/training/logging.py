"""Machine-readable run logging: append rows to a CSV (header written on first use),
so a crashed sweep keeps every completed run."""

import csv
from pathlib import Path


def append_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    new = not Path(path).exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)
