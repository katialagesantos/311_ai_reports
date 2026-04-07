"""
data_notes.py
Loads the 311 dataset once and reports, for each pipeline step,
exactly how many rows are included, excluded, and why.

Output: docs/data_notes.html  (self-contained)
"""
import argparse
import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ── config ─────────────────────────────────────────────────────────────────────
CSV_PATH      = "dataset/311_Service_Requests_20260329.csv"
WEATHER_CACHE = Path("aea/output/weather_calgary_daily.csv")
DEFAULT_OUT   = "docs/data_notes.html"

# Community map default sample cap (matches map_311_communities_v2.py)
MAP_POINTS_PER_YEAR = 3_000

# Calgary bounding box (matches weather_311_analysis.py and outlier_report.py)
LAT_MIN, LAT_MAX = 50.84, 51.21
LON_MIN, LON_MAX = -114.32, -113.86


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Generate a data-notes report showing row exclusions per pipeline step."
    )
    p.add_argument("--input",  default=CSV_PATH)
    p.add_argument("--output", default=DEFAULT_OUT)
    p.add_argument("--nrows",  type=int, default=None,
        help="Limit rows loaded (useful for quick tests).")
    return p.parse_args()


# ── helpers ────────────────────────────────────────────────────────────────────
def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def pct_str(count: int, total: int) -> str:
    """Format count as '1,234 (5.6%)' relative to total."""
    if total == 0:
        return f"{count:,}"
    return f"{count:,} &nbsp;<span class='pct'>({count / total * 100:.1f}%)</span>"


# ── data loading ───────────────────────────────────────────────────────────────
def load_data(csv_path: str, nrows) -> pd.DataFrame:
    cols = ["requested_date", "closed_date", "latitude", "longitude", "comm_name"]
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, usecols=cols, low_memory=False, nrows=nrows)
    print(f"  {len(df):,} rows loaded")
    return df


# ── per-step exclusion calculations ───────────────────────────────────────────
def compute_steps(df: pd.DataFrame) -> list[dict]:
    total = len(df)

    # ── Parse dates once ────────────────────────────────────────────────────
    req = pd.to_datetime(df["requested_date"], errors="coerce", format="mixed")
    cls = pd.to_datetime(df["closed_date"],    errors="coerce", format="mixed")

    n_missing_req = int(req.isna().sum())
    n_missing_cls = int(cls.isna().sum())

    days     = (cls - req).dt.total_seconds() / 86400
    has_both = req.notna() & cls.notna()
    n_negative  = int((has_both & (days < 0)).sum())
    n_valid_res = int((has_both & (days >= 0)).sum())

    # ── Coordinates ─────────────────────────────────────────────────────────
    has_coords = df["latitude"].notna() & df["longitude"].notna()
    n_no_coords = int((~has_coords).sum())

    has_coord_and_date    = has_coords & req.notna()
    n_no_date_with_coords = int((has_coords & req.isna()).sum())
    n_mappable            = int(has_coord_and_date.sum())

    # Map per-year sampling
    year_series = req[has_coord_and_date].dt.year.dropna()
    map_sample_total = sum(
        min(int(cnt), MAP_POINTS_PER_YEAR)
        for cnt in year_series.value_counts().values
    )
    n_map_downsampled = n_mappable - map_sample_total

    # ── Weather join ────────────────────────────────────────────────────────
    n_outside_weather = 0
    weather_range_str = "weather date range unknown"
    if WEATHER_CACHE.exists():
        wdf  = pd.read_csv(WEATHER_CACHE, parse_dates=["date"])
        wmin = wdf["date"].min()
        wmax = wdf["date"].max()
        weather_range_str = f"{wmin.date()} to {wmax.date()}"
        req_norm = req.dt.normalize()
        n_outside_weather = int(
            ((req_norm < wmin) | (req_norm > wmax)).sum()
        )

    # ── Chart 5c: Unknown community with no Calgary coords ─────────────────
    has_comm = df["comm_name"].notna() & (df["comm_name"].str.strip() != "")
    in_calgary = (
        df["latitude"].between(LAT_MIN, LAT_MAX)
        & df["longitude"].between(LON_MIN, LON_MAX)
    )
    n_5c_dropped = int((~has_comm & ~in_calgary & req.notna()).sum())

    # ── Assemble step descriptors ────────────────────────────────────────────
    steps = []

    # Step 1 – Dataset Profile
    steps.append({
        "num": "1",
        "report": "Dataset Profile",
        "script": "datasetProfile.py",
        "output": "report.html",
        "pct_used": 100.0,
        "rows": [
            ("Rows loaded",   total, "All rows"),
            ("Rows excluded", 0,     "No filtering — full dataset is profiled"),
        ],
        "note": (
            "ydata-profiling scans every row and column to compute statistics, "
            "correlations, and distributions. No rows are removed."
        ),
    })

    # Step 2 – Summary Analysis
    pct_res = n_valid_res / total * 100 if total else 100.0
    steps.append({
        "num": "2",
        "report": "Summary Analysis",
        "script": "analyze_311_summary.py",
        "output": "aea/output/generated_json/summary_full.json",
        "pct_used": pct_res,
        "rows": [
            ("Rows loaded", total, "All rows"),
            ("Missing requested_date",               n_missing_req, "Cannot parse date → excluded from all date-based metrics"),
            ("Missing closed_date",                  n_missing_cls, "Still-open requests → no resolution time computed"),
            ("Negative resolution days (closed < opened)", n_negative, "Data error → excluded from resolution stats"),
            ("Rows used for resolution stats",       n_valid_res,   "Both dates present and closed ≥ opened"),
        ],
        "note": (
            "Categorical counts and date-range summaries use all rows. "
            "Only the resolution-time statistics require both dates to be valid "
            "and non-negative."
        ),
    })

    # Step 3 – Chart Dashboard
    pct_chart = (total - n_missing_req) / total * 100 if total else 100.0
    steps.append({
        "num": "3",
        "report": "Chart Dashboard",
        "script": "plot_311_summary.py",
        "output": "summary_dashboard.html",
        "pct_used": pct_chart,
        "rows": [
            ("Rows loaded", total, "All rows"),
            ("Monthly volume chart — excluded",    n_missing_req,         "Unparseable requested_date → cannot assign to a month"),
            ("Resolution distribution — excluded", total - n_valid_res,   "Missing dates OR closed_date < requested_date"),
            ("Open/closed chart — excluded",       n_missing_req,         "Unparseable requested_date → cannot assign to a month"),
        ],
        "note": (
            "Each chart independently excludes only the rows it needs. "
            "Service-type counts are computed with fillna('MISSING'), "
            "so no rows are lost there."
        ),
    })

    # Step 4 – Community Map
    pct_map = n_mappable / total * 100 if total else 100.0
    steps.append({
        "num": "4",
        "report": "Community Map",
        "script": "map_311_communities_v2.py",
        "output": "community_map.html",
        "pct_used": pct_map,
        "rows": [
            ("Rows loaded", total, "All rows"),
            ("Missing latitude or longitude",               n_no_coords,          "No coordinates → cannot place pin on map"),
            ("Missing requested_date (among coord rows)",   n_no_date_with_coords, "Cannot determine year → excluded from year filter"),
            ("Rows with valid coordinates + date",          n_mappable,           "These feed the map before sampling"),
            (f"Downsampled (capped at {MAP_POINTS_PER_YEAR:,} per year)", n_map_downsampled,  "Random sample per year to keep HTML file size manageable in browser"),
            ("Points actually rendered on map",             map_sample_total,     "Final data shown interactively"),
        ],
        "note": (
            f"Each calendar year is independently sampled to at most "
            f"{MAP_POINTS_PER_YEAR:,} random points (default). "
            "This keeps the embedded HTML file small enough to load smoothly in a browser. "
            "The service-type distribution of the sample mirrors the full-year distribution."
        ),
    })

    # Step 5 – Weather Analysis
    n_weather_excl = n_missing_req + n_outside_weather
    pct_weather = max(0.0, (total - n_weather_excl) / total * 100) if total else 100.0
    steps.append({
        "num": "5",
        "report": "Weather Analysis",
        "script": "weather_311_analysis.py",
        "output": "weather_311_analysis.html",
        "pct_used": pct_weather,
        "rows": [
            ("Rows loaded", total, "All rows"),
            ("Missing/invalid requested_date",          n_missing_req,     "Cannot join to weather data by date"),
            ("Date outside weather archive coverage",   n_outside_weather, f"Weather data covers {weather_range_str}"),
            ("Chart 5c — unknown community + no Calgary coordinates", n_5c_dropped,
             "Nearest-centroid resolution requires a valid lat/lon within the Calgary bounding box"),
        ],
        "note": (
            "An inner join with the weather archive drops any 311 request whose date "
            "falls outside the available weather history. "
            "Charts 5a and 5b include all joined rows. "
            "Chart 5c uses the same joined rows but additionally resolves "
            "'Unknown' community rows via their coordinates — only rows with "
            "neither a community name nor valid Calgary coordinates are dropped."
        ),
    })

    # Step 6 – Outlier Report
    steps.append({
        "num": "6",
        "report": "Outlier & Anomaly Report",
        "script": "outlier_report.py",
        "output": "outlier_report.html",
        "pct_used": 100.0,
        "rows": [
            ("Rows loaded",   total, "All rows"),
            ("Rows excluded", 0,     "No filtering — all rows are analysed for anomalies"),
        ],
        "note": (
            "The outlier report intentionally processes all rows to surface data-quality "
            "issues. The anomalies it flags (negative resolution times, future dates, "
            "bad coordinates) are the same rows that other pipeline steps silently exclude."
        ),
    })

    return steps


# ── overview bar chart ────────────────────────────────────────────────────────
def overview_chart(steps: list[dict]) -> str:
    labels = [f"Step {s['num']} – {s['report']}" for s in steps]
    pcts   = [s["pct_used"] for s in steps]
    colors = [
        "#27ae60" if p >= 90 else
        "#e67e22" if p >= 70 else
        "#e74c3c"
        for p in pcts
    ]

    fig, ax = plt.subplots(figsize=(9, 0.55 * len(labels) + 1.4))
    bars = ax.barh(range(len(labels)), pcts, color=colors, edgecolor="white", height=0.55)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("% of total rows usable for this step", fontsize=9)
    ax.set_xlim(0, 112)
    ax.set_title("Data Utilisation per Pipeline Step", fontsize=11)

    for bar, pct in zip(bars, pcts):
        ax.text(
            pct + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center", fontsize=9,
        )

    ax.axvline(100, color="#bbb", linewidth=1, linestyle="--")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return fig_to_b64(fig)


# ── HTML builder ──────────────────────────────────────────────────────────────
def build_html(steps: list[dict], total: int, run_date: str) -> str:
    chart_b64 = overview_chart(steps)

    step_cards_html = ""
    for s in steps:
        rows_html = ""
        for label, count, reason in s["rows"]:
            if label == "Rows loaded":
                count_cell = f"{count:,}"
            elif count == 0:
                count_cell = "0 &nbsp;<span class='pct'>(none)</span>"
            else:
                count_cell = pct_str(count, total)
            rows_html += (
                f"<tr>"
                f"<td>{label}</td>"
                f"<td class='num'>{count_cell}</td>"
                f"<td class='reason'>{reason}</td>"
                f"</tr>"
            )

        step_cards_html += f"""
<div class="step-card">
  <div class="step-header">
    <span class="badge">Step {s["num"]}</span>
    <span class="step-name">{s["report"]}</span>
    <span class="step-meta">{s["script"]} &rarr; <code>{s["output"]}</code></span>
  </div>
  <table>
    <thead><tr><th>Category</th><th>Rows</th><th>Reason</th></tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
  <p class="step-note">{s["note"]}</p>
</div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Calgary 311 &#8212; Data Notes</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #f7f8fa; color: #1f2937; }}
  header {{ background: #0b3c5d; color: white; padding: 24px 32px; }}
  header h1 {{ font-size: 1.4rem; margin-bottom: 6px; }}
  header p  {{ opacity: .75; font-size: .88rem; }}
  main {{ max-width: 960px; margin: 32px auto; padding: 0 16px 64px; }}
  .chart-card {{
    background: white; border-radius: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,.1); padding: 20px; margin-bottom: 24px;
  }}
  .chart-card h2 {{ font-size: .95rem; color: #555; margin-bottom: 12px; }}
  .chart-card img {{ width: 100%; }}
  .step-card {{
    background: white; border-radius: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,.1); padding: 20px 24px; margin-bottom: 18px;
  }}
  .step-header {{
    display: flex; align-items: baseline; gap: 10px;
    flex-wrap: wrap; margin-bottom: 14px;
  }}
  .badge {{
    background: #0b3c5d; color: white; border-radius: 4px;
    padding: 2px 10px; font-size: .78rem; font-weight: bold; white-space: nowrap;
  }}
  .step-name {{ font-size: 1.05rem; font-weight: 600; }}
  .step-meta {{ font-size: .8rem; color: #777; }}
  .step-meta code {{ background: #f0f0f0; padding: 1px 5px; border-radius: 3px; font-size: .85em; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .875rem; margin-bottom: 12px; }}
  thead th {{
    background: #f0f4f8; padding: 7px 10px; text-align: left;
    border-bottom: 2px solid #dde3ea; color: #444; font-weight: 600;
  }}
  tbody td {{ padding: 6px 10px; border-bottom: 1px solid #f0f0f5; vertical-align: top; }}
  td.num {{ font-variant-numeric: tabular-nums; white-space: nowrap; font-weight: 500; }}
  .pct {{ color: #aaa; font-weight: normal; font-size: .85em; }}
  td.reason {{ color: #666; font-size: .85em; }}
  .step-note {{
    font-size: .84rem; color: #555;
    background: #f6f8fb; border-left: 3px solid #0b3c5d;
    padding: 8px 12px; border-radius: 0 4px 4px 0;
  }}
  a.back {{
    display: inline-block; margin-top: 28px; color: #0b3c5d;
    font-size: .9rem; text-decoration: none;
  }}
  a.back:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<header>
  <h1>&#128202; Data Notes &mdash; Row Exclusions per Pipeline Step</h1>
  <p>Total dataset rows: {total:,} &nbsp;&middot;&nbsp; Generated: {run_date}</p>
</header>
<main>
  <div class="chart-card">
    <h2>Data Utilisation Overview</h2>
    <img src="data:image/png;base64,{chart_b64}" alt="utilisation chart" />
  </div>
  {step_cards_html}
  <a class="back" href="index.html">&#8592; Back to index</a>
</main>
</body>
</html>"""


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    df   = load_data(args.input, args.nrows)
    total = len(df)

    print("Computing exclusion stats per pipeline step ...")
    steps = compute_steps(df)

    run_date = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    html     = build_html(steps, total, run_date)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    size_kb = out.stat().st_size / 1024
    print(f"Report saved to: {out}  ({size_kb:.1f} KB)")

    print("\nSummary:")
    for s in steps:
        print(f"  Step {s['num']}  {s['report']:<30s}  {s['pct_used']:5.1f}% usable")


if __name__ == "__main__":
    main()
