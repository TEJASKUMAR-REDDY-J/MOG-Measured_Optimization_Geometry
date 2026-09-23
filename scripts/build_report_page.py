"""Inline results/processed/report_data.json + reports/text.json into reports/atlas_report.html.

    python -m scripts.build_report_data && python -m scripts.build_report_page
"""

import json

from mog.utils.reproducibility import REPO_ROOT


def main():
    data = json.loads((REPO_ROOT / "results/processed/report_data.json").read_text(encoding="utf-8"))
    data["text"] = json.loads((REPO_ROOT / "reports/text.json").read_text(encoding="utf-8"))
    page = (REPO_ROOT / "reports/template.html").read_text(encoding="utf-8")
    blob = json.dumps(data, allow_nan=False).replace("</", "<\\/")  # keep </script> out of the JSON island
    out = REPO_ROOT / "reports/atlas_report.html"
    out.write_text(page.replace("/*DATA*/", blob), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
