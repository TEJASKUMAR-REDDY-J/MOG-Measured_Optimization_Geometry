"""Wall-clock timing for overhead measurements (NS, candidate evaluation, logging)."""

import time
from collections import defaultdict
from contextlib import contextmanager


class Timer:
    def __init__(self):
        self.total = defaultdict(float)
        self.count = defaultdict(int)

    @contextmanager
    def __call__(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.total[name] += time.perf_counter() - t0
            self.count[name] += 1

    def summary(self) -> dict:
        return {k: {"total_s": v, "calls": self.count[k]} for k, v in self.total.items()}
