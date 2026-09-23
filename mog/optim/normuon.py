"""NorMuon (arXiv:2510.05491): Newton-Schulz orthogonalization, then per-neuron (row-wise)
second-moment normalization and RMS matching:

    O = NS5(M);  v <- b2 v + (1 - b2) mean_cols(O * O);  O_hat = O / (sqrt(v) + eps);  RMS -> 0.2

Implemented as rule "normuon" of `BlockOptimizer` (blockwise.py, `direction`), so it can be
assigned per block and switched mid-run like every other rule.
"""

from .blockwise import BlockOptimizer

RULE = "normuon"
__all__ = ["BlockOptimizer", "RULE"]
