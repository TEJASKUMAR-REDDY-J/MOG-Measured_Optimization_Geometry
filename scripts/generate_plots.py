"""Cross-run figures. Each experiment writes its own figures/ at run time; the combined
interactive results page (reports/atlas_report.html) is built by:

    python -m scripts.generate_plots
"""

from reports import build_report_data, build_report_page

if __name__ == "__main__":
    build_report_data.main()
    build_report_page.main()
