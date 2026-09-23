"""MOG online geometry selector (PDF §5). NOT IMPLEMENTED: gated on evidence.

Per docs/RESEARCH_PLAN.md §4 the online selector is built only if Exp 2
(exp200_oracle_1m) shows the proxy E_c predicts the oracle (H2 not falsified).
Until then the evidence-backed MOG arm is the oracle-scheduled assignment
(`mog_oracle` in experiments/sweep.py), which applies measured per-block switches.
"""


class MOGOptimizer:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("online MOG selector is gated on H2 (Exp 2); see docs/RESEARCH_PLAN.md")
