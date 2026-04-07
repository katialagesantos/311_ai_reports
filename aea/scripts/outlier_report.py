"""
outlier_report.py
Scans the Calgary 311 dataset for anomalies and outliers across all
dimensions used by the reporting pipeline, then writes a self-contained
HTML report to docs/outlier_report.html.

Checks performed
----------------
1.  Resolution time outliers   — requests closed too fast (< 0 h) or
                                  extremely slow (> p99 + 3×IQR fence)
2.  Monthly volume spikes/dips — months whose volume deviates more than
                                  3 standard deviations from a 12-month
                                  rolling mean (Z-score on residuals)
3.  Date anomalies             — closed_date < requested_date,
                                  requested_date or closed_date in the future
4.  Geographic outliers        — lat/lon outside the Calgary bounding box
                                  (or present but clearly erroneous: 0,0 etc.)
5.  Service concentration      — service types whose share of requests in
                                  one weather/month bucket is > 3 SD above
                                  their average share (needs weather cache)
6.  Duplicate requests         — exact-duplicate rows in the dataset

Dependencies: pandas, numpy, matplotlib
"""
import argparse
import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── config ────────────────────────────────────────────────────────────────────
CSV_PATH      = "dataset/311_Service_Requests_20260329.csv"
WEATHER_CACHE = Path("aea/output/weather_calgary_daily.csv")
DEFAULT_OUT   = "docs/outlier_report.html"

# Calgary bounding box for coordinate checks
LAT_MIN, LAT_MAX = 50.84, 51.21
LON_MIN, LON_MAX = -114.32, -113.86

TODAY = pd.Timestamp.now().normalize()


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Generate an outlier / anomaly report for the Calgary 311 dataset."
    )
    p.add_argument("--input",   default=CSV_PATH)
    p.add_argument("--output",  default=DEFAULT_OUT)
    p.add_argument("--nrows",   type=int, default=None,
        help="Limit rows loaded (useful for quick tests).")
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────
def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def iqr_fence(series: pd.Series, k: float = 3.0):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def section(title: str, body: str, flag: str = "info") -> str:
    colours = {"ok": "#27ae60", "warn": "#e67e22", "error": "#e74c3c", "info": "#2980b9"}
    colour  = colours.get(flag, colours["info"])
    return f"""
<div class="section">
  <h2><span class="dot" style="background:{colour}"></span>{title}</h2>
  {body}
</div>"""


def stat_row(label: str, value: str) -> str:
    return f"<tr><td>{label}</td><td class='val'>{value}</td></tr>"


def table(rows: list[tuple], headers: list[str]) -> str:
    ths = "".join(f"<th>{h}</th>" for h in headers)
    trs = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{ths}</tr></thead><tbody>{trs}</tbody></table>"


def small_table(stats: dict) -> str:
    rows = "".join(stat_row(k, v) for k, v in stats.items())
    return f"<table class='stat-table'>{rows}</table>"


# ── checks ────────────────────────────────────────────────────────────────────

def check_resolution_time(df: pd.DataFrame) -> tuple[str, str]:
    """Check 1: resolution time outliers."""
    req = pd.to_datetime(df["requested_date"], errors="coerce", format="mixed")
    cls = pd.to_datetime(df["closed_date"],    errors="coerce", format="mixed")
    days = (cls - req).dt.total_seconds() / 86400
    has_both = req.notna() & cls.notna()

    negative   = has_both & (days < 0)
    fence_low, fence_high = iqr_fence(days[has_both & (days >= 0)])
    extreme_hi = has_both & (days > fence_high)

    n_neg  = int(negative.sum())
    n_hi   = int(extreme_hi.sum())
    p99    = days[has_both].quantile(0.99)
    median = days[has_both].quantile(0.50)

    # Distribution chart (0–p99)
    valid = days[has_both & (days >= 0)]
    clipped = valid[valid <= valid.quantile(0.99)]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(clipped, bins=60, color="#2980b9", alpha=0.75)
    ax.axvline(fence_high, color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"Outlier fence (3×IQR): {fence_high:.0f}d")
    ax.axvline(median, color="#27ae60", linestyle="--", linewidth=1.5,
               label=f"Median: {median:.1f}d")
    ax.set_xlabel("Days to close")
    ax.set_ylabel("Requests")
    ax.set_title("Resolution Time Distribution (0–p99)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    chart = fig_to_b64(fig)

    flag = "error" if (n_neg > 0 or n_hi > 1000) else "warn" if n_hi > 0 else "ok"
    stats = {
        "Pairs with both dates":   f"{int(has_both.sum()):,}",
        "Negative durations":      f"{n_neg:,}",
        "Above outlier fence":     f"{n_hi:,}  (fence = {fence_high:.0f} days)",
        "Median resolution":       f"{median:.1f} days",
        "p99 resolution":          f"{p99:.1f} days",
    }
    body = small_table(stats)
    if n_neg:
        body += f"<p class='note warn-note'>⚠ {n_neg:,} rows have closed_date < requested_date — likely data entry errors.</p>"
    if n_hi:
        body += f"<p class='note warn-note'>⚠ {n_hi:,} rows exceed the 3×IQR outlier fence of {fence_high:.0f} days.</p>"
    body += f'<img src="data:image/png;base64,{chart}" style="max-width:100%;margin-top:14px;border-radius:6px;">'
    return section("1. Resolution Time Outliers", body, flag), flag


def check_monthly_volume(df: pd.DataFrame) -> tuple[str, str]:
    """Check 2: monthly volume spikes and dips (Z-score on rolling residuals)."""
    req = pd.to_datetime(df["requested_date"], errors="coerce", format="mixed")
    monthly = req.dt.to_period("M").value_counts().sort_index()
    monthly.index = monthly.index.to_timestamp()
    monthly = monthly.sort_index()

    rolling_mean = monthly.rolling(12, center=True, min_periods=3).mean()
    residuals    = monthly - rolling_mean
    z_scores     = (residuals - residuals.mean()) / residuals.std()

    spikes = monthly[z_scores.abs() > 3].sort_index()

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(monthly.index, monthly.values, color="#2980b9", linewidth=1.5, label="Monthly volume")
    ax.plot(monthly.index, rolling_mean.values, color="#888", linewidth=1, linestyle="--",
            label="12-month rolling mean")
    if not spikes.empty:
        ax.scatter(spikes.index, spikes.values, color="#e74c3c", zorder=5,
                   s=60, label="Outlier (|Z|>3)")
    ax.set_title("Monthly Request Volume with Outliers Highlighted")
    ax.set_ylabel("Requests")
    ax.legend(fontsize=9)
    fig.tight_layout()
    chart = fig_to_b64(fig)

    flag = "warn" if not spikes.empty else "ok"
    stats = {
        "Months analysed":    f"{len(monthly):,}",
        "Outlier months":     f"{len(spikes):,}",
        "Peak month":         f"{monthly.idxmax().strftime('%Y-%m')} ({int(monthly.max()):,} requests)",
        "Lowest month":       f"{monthly.idxmin().strftime('%Y-%m')} ({int(monthly.min()):,} requests)",
    }
    body = small_table(stats)
    if not spikes.empty:
        rows = [(m.strftime("%Y-%m"), f"{int(v):,}", f"{z_scores[m]:+.1f}")
                for m, v in spikes.items()]
        body += "<p class='note warn-note'>⚠ Months with |Z-score| > 3 vs rolling mean:</p>"
        body += table(rows, ["Month", "Requests", "Z-score"])
    body += f'<img src="data:image/png;base64,{chart}" style="max-width:100%;margin-top:14px;border-radius:6px;">'
    return section("2. Monthly Volume Spikes & Dips", body, flag), flag


def check_date_anomalies(df: pd.DataFrame) -> tuple[str, str]:
    """Check 3: closed < requested, future dates."""
    req = pd.to_datetime(df["requested_date"], errors="coerce", format="mixed")
    cls = pd.to_datetime(df["closed_date"],    errors="coerce", format="mixed")

    future_req = int((req > TODAY).sum())
    future_cls = int((cls > TODAY).sum())
    closed_before_opened = int((cls < req).sum())
    req_missing = int(req.isna().sum())
    cls_missing = int(cls.isna().sum())

    flag = "error" if closed_before_opened > 0 or future_req > 0 else "ok"
    stats = {
        "requested_date missing":       f"{req_missing:,}  ({req_missing/len(df)*100:.1f}%)",
        "closed_date missing":          f"{cls_missing:,}  ({cls_missing/len(df)*100:.1f}%)",
        "closed_date < requested_date": f"{closed_before_opened:,}",
        "Future requested_date":        f"{future_req:,}",
        "Future closed_date":           f"{future_cls:,}",
    }
    body = small_table(stats)
    if closed_before_opened:
        body += f"<p class='note error-note'>✖ {closed_before_opened:,} rows have closed_date before requested_date.</p>"
    if future_req:
        body += f"<p class='note error-note'>✖ {future_req:,} rows have a requested_date in the future.</p>"
    if flag == "ok":
        body += "<p class='note ok-note'>✔ No critical date anomalies detected.</p>"
    return section("3. Date Anomalies", body, flag), flag


def check_coordinates(df: pd.DataFrame) -> tuple[str, str]:
    """Check 4: lat/lon outside Calgary bounding box or zero-island."""
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return section("4. Geographic Outliers", "<p>latitude/longitude columns not found.</p>", "info"), "info"

    lat = pd.to_numeric(df["latitude"],  errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    has_coords = lat.notna() & lon.notna()

    zero_island  = has_coords & (lat.abs() < 1) & (lon.abs() < 1)
    out_of_bbox  = has_coords & ~zero_island & (
        ~lat.between(LAT_MIN, LAT_MAX) | ~lon.between(LON_MIN, LON_MAX)
    )

    n_zero  = int(zero_island.sum())
    n_bbox  = int(out_of_bbox.sum())
    n_valid = int((has_coords & ~zero_island & ~out_of_bbox).sum())
    n_missing = int((~has_coords).sum())

    # Scatter: valid + outliers
    fig, ax = plt.subplots(figsize=(9, 8))
    valid_mask = has_coords & ~zero_island & ~out_of_bbox
    sample_size = min(50_000, int(valid_mask.sum()))
    sample = df[valid_mask].sample(sample_size, random_state=42) if sample_size > 0 else df[valid_mask]
    ax.scatter(lon[sample.index], lat[sample.index],
               s=0.5, alpha=0.15, color="#2980b9", label="Valid")
    if n_bbox > 0:
        ax.scatter(lon[out_of_bbox], lat[out_of_bbox],
                   s=10, color="#e74c3c", label=f"Outside bbox ({n_bbox:,})", zorder=5)
    ax.set_xlim(LON_MIN - 0.1, LON_MAX + 0.1)
    ax.set_ylim(LAT_MIN - 0.05, LAT_MAX + 0.05)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Request Coordinates (Calgary bounding box)")
    ax.legend(fontsize=9, markerscale=6)
    fig.tight_layout()
    chart = fig_to_b64(fig)

    flag = "warn" if (n_zero + n_bbox) > 0 else "ok"
    stats = {
        "Has coordinates":             f"{int(has_coords.sum()):,}",
        "Missing coordinates":         f"{n_missing:,}  ({n_missing/len(df)*100:.1f}%)",
        "Valid (within Calgary bbox)": f"{n_valid:,}",
        "Zero-island (lat≈0, lon≈0)":  f"{n_zero:,}",
        "Outside Calgary bbox":        f"{n_bbox:,}",
    }
    body = small_table(stats)
    if n_zero:
        body += f"<p class='note warn-note'>⚠ {n_zero:,} rows have coordinates near (0, 0) — likely missing data encoded as zero.</p>"
    if n_bbox:
        body += f"<p class='note warn-note'>⚠ {n_bbox:,} rows have coordinates outside the Calgary bounding box.</p>"
    if flag == "ok":
        body += "<p class='note ok-note'>✔ All coordinates fall within the expected Calgary bounding box.</p>"
    body += f'<img src="data:image/png;base64,{chart}" style="max-width:100%;margin-top:14px;border-radius:6px;">'
    return section("4. Geographic Outliers", body, flag), flag


def check_duplicates(df: pd.DataFrame) -> tuple[str, str]:
    """Check 5: duplicate rows."""
    n_dupes = int(df.duplicated().sum())
    flag = "warn" if n_dupes > 0 else "ok"
    stats = {
        "Total rows":       f"{len(df):,}",
        "Duplicate rows":   f"{n_dupes:,}  ({n_dupes/len(df)*100:.2f}%)",
        "Unique rows":      f"{len(df) - n_dupes:,}",
    }
    body = small_table(stats)
    if n_dupes:
        body += f"<p class='note warn-note'>⚠ {n_dupes:,} exact duplicate rows found.</p>"
    else:
        body += "<p class='note ok-note'>✔ No duplicate rows detected.</p>"
    return section("5. Duplicate Rows", body, flag), flag


def check_service_concentration(df: pd.DataFrame) -> tuple[str, str]:
    """Check 6: service types with abnormally high monthly concentration."""
    req = pd.to_datetime(df["requested_date"], errors="coerce", format="mixed")
    df2 = df.assign(ym=req.dt.to_period("M")).dropna(subset=["ym"])

    top_services = df2["service_name"].value_counts().head(20).index
    sub = df2[df2["service_name"].isin(top_services)]

    pivot = (
        sub.groupby(["ym", "service_name"])
        .size()
        .unstack(fill_value=0)
    )
    # Share of monthly total per service
    monthly_total = pivot.sum(axis=1)
    share = pivot.div(monthly_total, axis=0)

    # Z-score per service column
    z = (share - share.mean()) / share.std()
    outlier_cells = (z.abs() > 3).stack()
    outlier_cells = outlier_cells[outlier_cells]

    flag = "warn" if not outlier_cells.empty else "ok"
    stats = {
        "Services analysed":        f"{len(top_services)}",
        "Months analysed":          f"{len(pivot)}",
        "Concentration outliers":   f"{len(outlier_cells):,}  (service×month combinations with |Z|>3)",
    }
    body = small_table(stats)
    if not outlier_cells.empty:
        rows_out = []
        for (ym, svc), _ in outlier_cells.items():
            z_val = float(z.loc[ym, svc])
            share_val = float(share.loc[ym, svc]) * 100
            rows_out.append((str(ym), svc, f"{share_val:.1f}%", f"{z_val:+.1f}"))
        rows_out.sort(key=lambda r: -abs(float(r[3])))
        body += "<p class='note warn-note'>⚠ Service types with unusually high monthly share (|Z|>3):</p>"
        body += table(rows_out[:30], ["Month", "Service", "Monthly share", "Z-score"])
        if len(rows_out) > 30:
            body += f"<p class='note'>… and {len(rows_out)-30} more.</p>"
    else:
        body += "<p class='note ok-note'>✔ No unusual service concentration detected.</p>"
    return section("6. Service Concentration Outliers", body, flag), flag


# ── report ────────────────────────────────────────────────────────────────────
FLAG_LABEL = {"ok": "✔ OK", "warn": "⚠ Warning", "error": "✖ Error", "info": "ℹ Info"}
FLAG_COLOR = {"ok": "#27ae60", "warn": "#e67e22", "error": "#e74c3c", "info": "#2980b9"}


def build_report(sections: list[tuple[str, str]], n_rows: int) -> str:
    summary_rows = "".join(
        f"<tr><td>{title}</td>"
        f"<td style='color:{FLAG_COLOR[flag]};font-weight:bold'>{FLAG_LABEL[flag]}</td></tr>"
        for title, flag in sections
    )
    bodies = "\n".join(s for s, _ in sections)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Calgary 311 – Outlier Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: Arial, sans-serif; background: #f4f6f9; color: #222; }}
  header {{ background: #1e2a38; color: white; padding: 20px 32px; }}
  header h1 {{ font-size: 22px; margin-bottom: 4px; }}
  header p {{ color: #8ab; font-size: 13px; }}
  .section {{
    margin: 24px 32px; background: white;
    border-radius: 8px; padding: 20px 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,.08);
  }}
  .section h2 {{
    font-size: 16px; color: #2c3e50;
    margin-bottom: 14px; display: flex; align-items: center; gap: 8px;
  }}
  .dot {{
    display: inline-block; width: 12px; height: 12px;
    border-radius: 50%; flex-shrink: 0;
  }}
  .stat-table {{ border-collapse: collapse; font-size: 13px; margin-bottom: 12px; }}
  .stat-table td {{ padding: 4px 16px 4px 0; }}
  .stat-table td.val {{ font-weight: bold; color: #2c3e50; }}
  table {{ border-collapse: collapse; font-size: 12px; width: 100%; margin-top: 10px; }}
  th {{ background: #2c3e50; color: white; padding: 5px 10px; text-align: left; }}
  td {{ padding: 4px 10px; border-bottom: 1px solid #eee; }}
  tr:nth-child(even) {{ background: #f8f9fa; }}
  .note {{ font-size: 12px; margin-top: 10px; padding: 8px 12px; border-radius: 4px; background: #f8f9fa; }}
  .warn-note  {{ background: #fff8e1; color: #b26a00; }}
  .error-note {{ background: #ffeaea; color: #c0392b; }}
  .ok-note    {{ background: #eafaf1; color: #1e8449; }}
  .summary-table {{ border-collapse: collapse; font-size: 13px; width: 100%; max-width: 500px; }}
  .summary-table td {{ padding: 6px 12px; border-bottom: 1px solid #eee; }}
  footer {{ text-align: center; padding: 20px; color: #999; font-size: 12px; }}
</style>
</head>
<body>
<header>
  <h1>&#128269; Calgary 311 – Outlier &amp; Anomaly Report</h1>
  <p>Automated quality checks on {n_rows:,} rows from the 311 dataset</p>
</header>

<div class="section">
  <h2><span class="dot" style="background:#2980b9"></span>Summary</h2>
  <table class="summary-table">
    <tr><th>Check</th><th>Result</th></tr>
    {summary_rows}
  </table>
</div>

{bodies}

<footer>
  Data: City of Calgary 311 Open Data &nbsp;|&nbsp;
  Generated {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}
</footer>
</body>
</html>"""


# ── main ──────────────────────────────────────────────────────────────────────
CHECK_TITLES = [
    "1. Resolution Time Outliers",
    "2. Monthly Volume Spikes & Dips",
    "3. Date Anomalies",
    "4. Geographic Outliers",
    "5. Duplicate Rows",
    "6. Service Concentration Outliers",
]


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading 311 data from {args.input} ...")
    df = pd.read_csv(args.input, low_memory=False, nrows=args.nrows)
    print(f"  {len(df):,} rows loaded.")

    print("\nRunning outlier checks ...")
    results = []
    checks = [
        ("Resolution time", check_resolution_time),
        ("Monthly volume",  check_monthly_volume),
        ("Date anomalies",  check_date_anomalies),
        ("Coordinates",     check_coordinates),
        ("Duplicates",      check_duplicates),
        ("Service concentration", check_service_concentration),
    ]
    for name, fn in checks:
        print(f"  [{name}] ...", end=" ", flush=True)
        sec_html, flag = fn(df)
        results.append((sec_html, flag))
        print(FLAG_LABEL[flag])

    summary = list(zip(CHECK_TITLES, [flag for _, flag in results]))
    html = build_report(list(zip([s for s, _ in results], [f for _, f in results])), len(df))
    output_path.write_text(html, encoding="utf-8")

    size_mb = output_path.stat().st_size / 1_048_576
    print(f"\nReport saved to: {output_path}  ({size_mb:.1f} MB)")

    print("\nSummary:")
    for title, flag in summary:
        print(f"  {FLAG_LABEL[flag]:12s}  {title}")


if __name__ == "__main__":
    main()
