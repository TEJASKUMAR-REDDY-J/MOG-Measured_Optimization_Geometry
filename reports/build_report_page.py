"""Inline reports/report_data.json + reports/text.json into reports/atlas_report.html.

    python -m reports.build_report_data && python -m reports.build_report_page
"""

import json

from mog.utils.reproducibility import REPO_ROOT


def main():
    text = REPO_ROOT / "reports/text.json"
    if not text.exists():  # ponytail: page UI paused; data build still works without it
        print("skipped page: reports/text.json (findings text) not written yet")
        return
    data = json.loads((REPO_ROOT / "reports/report_data.json").read_text(encoding="utf-8"))
    data["text"] = json.loads(text.read_text(encoding="utf-8"))
    page = (REPO_ROOT / "reports/template.html").read_text(encoding="utf-8")
    blob = json.dumps(data, allow_nan=False).replace("</", "<\\/")  # keep </script> out of the JSON island
    out = REPO_ROOT / "reports/atlas_report.html"
    out.write_text(page.replace("/*DATA*/", blob), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
