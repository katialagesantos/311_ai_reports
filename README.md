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

## Weather Analysis Report (weather_311_analysis.py)

Generates `docs/weather_311_analysis.html` — a self-contained report correlating Calgary 311
service requests with daily weather conditions from the Open-Meteo historical archive.

### Charts included

| Section | Description |
|---------|-------------|
| 1 | Spearman correlation heatmap: service types vs weather variables |
| 2 | Seasonal distribution heatmap by service type |
| 3 | Average daily requests by weather category |
| 4 | Monthly request volume vs average max temperature |
| 5a | Top 15 communities during extreme weather (all requests, including Unknown) |
| 5b | Top 15 communities during extreme weather (assigned communities only) |
| 5c | Top 15 communities during extreme weather (Unknown rows resolved via lat/lon) |
| 6 | Service type spike ratios during each extreme weather category |

### How chart 5c resolves "Unknown" communities

Some 311 requests have no `comm_name` recorded. Chart 5c uses the request's `latitude`
and `longitude` to assign it to the nearest known community via a **nearest centroid** lookup:

1. **Build centroids** — for every row that already has a `comm_name`, compute the average
   latitude and longitude for that community across all its extreme-weather requests.
   This gives one representative coordinate per community (e.g. "BELTLINE" → 51.04°N, 114.08°W).

2. **Find the closest one** — for each "Unknown" row, calculate the Euclidean distance in
   lat/lon space to every community centroid, then assign the community whose centroid is nearest.

**Caveats:**
- Assigns to the nearest *centroid of requests*, not the official Calgary community boundary polygon.
  A request on the border between two communities may be assigned to the wrong one.
- Centroids are computed from extreme-weather requests only, so they may drift slightly from
  the true geographic centre of each community.
- Uses flat Euclidean distance on lat/lon degrees (not geodesic). At Calgary's latitude the
  error is under 0.1% across the ~35 km city span — accurate enough for this analysis.
- Requests with neither a community name nor valid coordinates within the Calgary bounding box
  are excluded entirely.

For higher precision, a point-in-polygon test against the official Calgary community boundary
GeoJSON (via `geopandas`) would be required.
