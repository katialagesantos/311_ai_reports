# Calgary 311 Analysis Pipeline

## AEA - Automated Exploratory Analysis

This workspace analyzes Calgary 311 service requests and generates:
- A profiling report HTML
- A numeric summary JSON and console report
- Charts and a dashboard HTML

## Data Input

Default CSV input:
- dataset/311_Service_Requests_20260329.csv

## Main Scripts

- aea/scripts/datasetProfile.py
  - Builds a ydata-profiling HTML report.

- aea/scripts/analyze_311_summary.py
  - Prints summary stats and can save JSON output.

- aea/scripts/plot_311_summary.py
  - Generates chart images and summary_dashboard.html.

- aea/scripts/run_pipeline_311_reports.py
  - Runs the full pipeline in order.

## Pipeline Command

Run everything:

venv/bin/python aea/scripts/run_pipeline_311_reports.py

Run everything on the full dataset with top 10 categories:

venv/bin/python aea/scripts/run_pipeline_311_reports.py --top-n 10

Run everything except profile generation:

venv/bin/python aea/scripts/run_pipeline_311_reports.py --skip-profile

## What --nrows Means

--nrows limits how many rows are read from the CSV for analysis and plotting steps.

Use it for faster testing.

Examples:
- --nrows 100000 reads only the first 100,000 rows.
- If omitted, scripts use the full dataset.

## What --top-n Means

--top-n controls how many top categories are shown in outputs that use rankings.

Examples:
- In top services charts, --top-n 10 shows top 10 service request types.
- --top-n 20 shows top 20.

## Useful Examples

Fast sample run:

venv/bin/python aea/scripts/run_pipeline_311_reports.py --skip-profile --nrows 300000 --top-n 10

Full run with profile:

venv/bin/python aea/scripts/run_pipeline_311_reports.py --top-n 10

Custom output locations:

venv/bin/python aea/scripts/run_pipeline_311_reports.py --analysis-json aea/output/generated_json/summary_full.json --charts-output-dir aea/output/charts --dashboard-html aea/reports/summary_dashboard.html

## Typical Outputs

- aea/reports/report.html
- aea/reports/summary_dashboard.html
- aea/output/generated_json/summary_full.json
- aea/output/charts/monthly_requests.png
- aea/output/charts/top_services.png
- aea/output/charts/resolution_days_distribution.png
- aea/output/charts/open_vs_closed_by_request_month.png
