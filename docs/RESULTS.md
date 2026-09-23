# RESULTS

Only measured results appear here. Each result links to its run directory under `results/raw/`. Status keys: **PASS**, **FAIL**, **INCONCLUSIVE**, **NOT TESTED**.

## Synthetic / theory checks (evidence class `synthetic`)

| Check | Status | Evidence |
|---|---|---|
| PDF §4.3 closed-form columns (r_eff; full-spectral θ/C) | PASS (unit test, closed form) | `tests/test_selection_metrics.py::test_pdf_table_closed_form_columns`. Exact values: 512, 281.65, 28.28, 5.30 and 1.000, 0.5501, 0.0552, 0.0104. |
| PDF §4.3 Monte Carlo columns (rows, elementwise) | NOT TESTED | Exp 0 pending |
| Fig. 1b / 1c trends | NOT TESTED | Exp 0 pending |
| NS nuclear-norm proxy fidelity (Prop. 2 assumption) | NOT TESTED | Exp 0 pending. The smoke run hints at a bias; see EXPERIMENT_LOG infra-000. |

## Hypotheses H1–H9

| # | Status | Evidence |
|---|---|---|
| H1 | NOT TESTED | — |
| H2 | NOT TESTED | — |
| H3 | NOT TESTED | — |
| H4 | NOT TESTED | — |
| H5 | NOT TESTED | — |
| H6 | NOT TESTED | — |
| H7 | NOT TESTED | — |
| H8 | NOT TESTED | — |
| H9 | NOT TESTED | — |

No training experiments have been run.
