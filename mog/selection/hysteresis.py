"""Hysteresis rule of PDF §5: switch only if the argmax beats the incumbent by a relative
margin delta on `patience` consecutive probes. Pure logic, independent of the gated selector."""

from __future__ import annotations


class Hysteresis:
    def __init__(self, margin: float, patience: int = 2):
        self.margin, self.patience = margin, patience
        self.pending: str | None = None
        self.count = 0

    def update(self, incumbent: str, scores: dict[str, float]) -> str:
        """Return the geometry to use after this probe."""
        best = max(scores, key=scores.get)
        if best != incumbent and scores[best] > scores[incumbent] * (1 + self.margin):
            self.count = self.count + 1 if best == self.pending else 1
            self.pending = best
            if self.count >= self.patience:
                self.pending, self.count = None, 0
                return best
        else:
            self.pending, self.count = None, 0
        return incumbent
