# reports/

This folder holds the interactive results page. It is a **work in progress and private**: it is not published and not linked from the docs.

| File | Contents |
|---|---|
| `template.html` | Page layout: network diagram coloured by geometry family, loss charts, oracle and proxy views, atlas and synthetic panels |
| `build_report_data.py` | Aggregates the latest run of each experiment into `report_data.json` (gitignored) |
| `build_report_page.py` | Inlines that data and `text.json` (the findings text, not written yet) into `atlas_report.html` (gitignored) |

Build it with:

```bash
python -m scripts.generate_plots
```

The canonical numbers always live in `results/raw/` and `docs/RESULTS.md`, never only here.
