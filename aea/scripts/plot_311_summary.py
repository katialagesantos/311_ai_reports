import argparse
import base64
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create chart outputs for Calgary 311 service requests."
    )
    parser.add_argument(
        "--input",
        default="dataset/311_Service_Requests_20260329.csv",
        help="Path to the 311 CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default="aea/output/charts",
        help="Directory where chart images will be written.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Top N service categories to plot.",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Optional row limit for quick iterations.",
    )
    parser.add_argument(
        "--html-output",
        default="docs/summary_dashboard.html",
        help="Output path for dashboard HTML file (default: eda/reports/summary_dashboard.html).",
    )
    return parser.parse_args()


def load_data(csv_path: Path, nrows: int | None) -> pd.DataFrame:
    needed_cols = ["requested_date", "closed_date", "service_name"]
    return pd.read_csv(csv_path, usecols=needed_cols, low_memory=False, nrows=nrows)


def plot_monthly_requests(df: pd.DataFrame, output_dir: Path) -> tuple[Path, dict[str, str]]:
    req = pd.to_datetime(df["requested_date"], errors="coerce", format="mixed")
    monthly = req.dt.to_period("M").value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(14, 6))
    monthly.index = monthly.index.to_timestamp()
    ax.plot(monthly.index, monthly.values, linewidth=2)
    ax.set_title("Monthly 311 Request Volume")
    ax.set_xlabel("Month")
    ax.set_ylabel("Requests")
    ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(alpha=0.25)

    out = output_dir / "monthly_requests.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

    metrics = {
        "Date range": f"{monthly.index.min().strftime('%Y-%m-%d')} to {monthly.index.max().strftime('%Y-%m-%d')}",
        "Peak month": f"{monthly.idxmax().strftime('%Y-%m-%d')} ({int(monthly.max()):,} requests)",
        "Average monthly volume": f"{monthly.mean():,.0f}",
    }
    return out, metrics


def plot_top_services(
    df: pd.DataFrame, output_dir: Path, top_n: int
) -> tuple[Path, dict[str, str]]:
    counts = (
        df["service_name"]
        .astype("string")
        .fillna("MISSING")
        .value_counts()
        .head(top_n)
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(counts.index, counts.values)
    ax.set_title(f"Top {top_n} Service Request Types")
    ax.set_xlabel("Request Count")
    ax.set_ylabel("Service Name")
    ax.grid(axis="x", alpha=0.25)

    out = output_dir / "top_services.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

    metrics = {
        "Top service": f"{counts.index[-1]} ({int(counts.iloc[-1]):,})",
        "Requests in top group": f"{int(counts.sum()):,}",
        "Share of all requests": f"{(counts.sum() / len(df)) * 100:.1f}%",
    }
    return out, metrics


def plot_resolution_distribution(
    df: pd.DataFrame, output_dir: Path
) -> tuple[Path, dict[str, str]]:
    req = pd.to_datetime(df["requested_date"], errors="coerce", format="mixed")
    cls = pd.to_datetime(df["closed_date"], errors="coerce", format="mixed")

    days = (cls - req).dt.total_seconds() / 86400
    valid = days[(days >= 0) & days.notna()]

    if valid.empty:
        raise ValueError("No valid requested_date/closed_date durations were found.")

    p95 = valid.quantile(0.95)
    clipped = valid[valid <= p95]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(clipped, bins=50)
    ax.set_title("Resolution Time Distribution (0 to 95th Percentile)")
    ax.set_xlabel("Days to Close")
    ax.set_ylabel("Number of Requests")
    ax.grid(alpha=0.25)

    for q in [0.5, 0.75, 0.9, 0.95]:
        value = valid.quantile(q)
        ax.axvline(value, linestyle="--", linewidth=1.5, label=f"p{int(q*100)}: {value:.1f}d")

    ax.legend()

    out = output_dir / "resolution_days_distribution.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

    metrics = {
        "Valid durations": f"{len(valid):,}",
        "Median (p50)": f"{valid.quantile(0.5):.1f} days",
        "p90": f"{valid.quantile(0.9):.1f} days",
        "p95": f"{valid.quantile(0.95):.1f} days",
    }
    return out, metrics


def plot_open_vs_closed_created_by_month(
    df: pd.DataFrame, output_dir: Path
) -> tuple[Path, dict[str, str]]:
    req = pd.to_datetime(df["requested_date"], errors="coerce", format="mixed")
    month = req.dt.to_period("M")

    status_bucket = pd.Series("Open", index=df.index)
    status_bucket[df["closed_date"].notna()] = "Closed"

    monthly = (
        pd.DataFrame({"month": month, "status": status_bucket})
        .dropna(subset=["month"])
        .groupby(["month", "status"])  # type: ignore[arg-type]
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    for col in ["Closed", "Open"]:
        if col not in monthly.columns:
            monthly[col] = 0

    fig, ax = plt.subplots(figsize=(14, 6))
    monthly_ts = monthly.copy()
    monthly_ts.index = monthly_ts.index.to_timestamp()
    ax.plot(monthly_ts.index, monthly_ts["Closed"], label="Closed tickets", linewidth=2)
    ax.set_title("Closed Volume and Open Share by Request Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Closed Requests")
    ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(alpha=0.25)

    total = monthly_ts["Closed"] + monthly_ts["Open"]
    open_share = (monthly_ts["Open"] / total.replace(0, pd.NA) * 100).fillna(0)
    ax2 = ax.twinx()
    ax2.plot(
        monthly_ts.index,
        open_share,
        color="#ef6c00",
        linewidth=1.8,
        label="Open share (%)",
    )
    ax2.set_ylabel("Open Share (%)")

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    out = output_dir / "open_vs_closed_by_request_month.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

    total = monthly["Closed"] + monthly["Open"]
    close_rate = (monthly["Closed"] / total.replace(0, pd.NA) * 100).mean()
    open_rate = (monthly["Open"] / total.replace(0, pd.NA) * 100).mean()
    metrics = {
        "Average closed share": f"{close_rate:.1f}%",
        "Average open share": f"{open_rate:.1f}%",
        "Peak open month": f"{monthly['Open'].idxmax()} ({int(monthly['Open'].max()):,})",
        "Peak closed month": f"{monthly['Closed'].idxmax()} ({int(monthly['Closed'].max()):,})",
    }
    return out, metrics


def build_service_counts_by_year(df: pd.DataFrame) -> tuple[list[int], dict[str, dict[str, int]]]:
    req = pd.to_datetime(df["requested_date"], errors="coerce", format="mixed")
    services = df["service_name"].astype("string").fillna("MISSING")

    yearly = pd.DataFrame({"year": req.dt.year, "service_name": services}).dropna(
        subset=["year"]
    )
    yearly["year"] = yearly["year"].astype(int)

    grouped = (
        yearly.groupby(["year", "service_name"])  # type: ignore[arg-type]
        .size()
        .reset_index(name="count")
    )

    years = sorted(grouped["year"].unique().tolist())
    counts_by_year: dict[str, dict[str, int]] = {}
    for year in years:
        part = grouped[grouped["year"] == year]
        counts_by_year[str(year)] = {
            str(row["service_name"]): int(row["count"]) for _, row in part.iterrows()
        }

    return years, counts_by_year


def render_html_dashboard(
    sections: list[dict[str, object]],
    html_path: Path,
    data_rows: int,
    input_file: Path,
    top_n: int,
    service_years: list[int],
    service_counts_by_year: dict[str, dict[str, int]],
    highlights: dict[str, str] | None = None,
) -> Path:
    section_html: list[str] = []
    for section in sections:
        title = html.escape(str(section["title"]))
        summary = html.escape(str(section["summary"]))
        image_src = ""
        if section.get("image"):
            img_path = Path(str(section["image"]))
            if img_path.exists():
                b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
                image_src = f"data:image/png;base64,{b64}"
        metrics = section.get("metrics", {})
        section_id = str(section.get("id", ""))

        metric_items = "\n".join(
            f"<li><strong>{html.escape(str(k))}:</strong> {html.escape(str(v))}</li>"
            for k, v in dict(metrics).items()
        )

        if section_id == "top_services":
            section_html.append(
                f"""
                <section class=\"card\" id=\"top-services-section\">
                  <h2>{title}</h2>
                  <p>{summary}</p>
                  <ul>
                    {metric_items}
                  </ul>
                                    <img id="top-services-image" src="{image_src}" alt="{title}" />
                  <div class=\"controls\">
                    <label for=\"start-year\">Start year</label>
                    <select id=\"start-year\"></select>
                    <label for=\"end-year\">End year</label>
                    <select id=\"end-year\"></select>
                  </div>
                  <p id=\"top-services-range\" class=\"range-label\"></p>
                                    <div id="top-services-canvas-wrap" style="display:none;">
                                        <canvas id="top-services-canvas"></canvas>
                                    </div>
                </section>
                """
            )
        else:
            section_html.append(
                f"""
                <section class=\"card\">
                  <h2>{title}</h2>
                  <p>{summary}</p>
                  <ul>
                    {metric_items}
                  </ul>
                  <img src=\"{image_src}\" alt=\"{title}\" />
                </section>
                """
            )

    highlights_items = ""
    if highlights:
        highlights_items = "\n".join(
            f"""          <div class="highlight-item">
            <span class="highlight-label">{html.escape(str(k))}</span>
            <span class="highlight-value">{html.escape(str(v))}</span>
          </div>"""
            for k, v in highlights.items()
        )
    highlights_card = f"""
        <section class="card highlights-card">
          <h2>Key Highlights</h2>
          <div class="highlights-grid">
{highlights_items}
          </div>
        </section>
    """ if highlights else ""

    years_json = json.dumps(service_years)
    counts_json = json.dumps(service_counts_by_year)

    html_doc = f"""<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Calgary 311 Chart Summary</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif;
            margin: 0;
            background: #f7f8fa;
            color: #1f2937;
        }}
        header {{
            background: #0b3c5d;
            color: #ffffff;
            padding: 24px;
        }}
        main {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            display: grid;
            gap: 20px;
        }}
        .card {{
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 16px;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        }}
        .controls {{
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
            margin: 8px 0 12px;
        }}
        .controls select {{
            padding: 6px 8px;
            border-radius: 6px;
            border: 1px solid #cbd5e1;
            background: #fff;
        }}
        .range-label {{
            margin: 0 0 8px;
            color: #334155;
            font-weight: 600;
        }}
        .highlights-card {{
            background: #eef4fb;
            border-color: #b6d0ea;
        }}
        .highlights-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 12px;
            margin-top: 10px;
        }}
        .highlight-item {{
            background: #ffffff;
            border: 1px solid #d1e3f5;
            border-radius: 8px;
            padding: 12px 14px;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}
        .highlight-label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #6b7280;
        }}
        .highlight-value {{
            font-size: 1rem;
            font-weight: 600;
            color: #0b3c5d;
        }}
        #top-services-canvas-wrap {{
            margin-top: 10px;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            background: #ffffff;
            padding: 12px;
        }}
        #top-services-canvas {{
            width: 100%;
            height: auto;
        }}
        img {{
            width: 100%;
            height: auto;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            margin-top: 10px;
        }}
        @media (max-width: 900px) {{
            .bar-row {{
                grid-template-columns: 1fr;
            }}
            .bar-value {{
                text-align: left;
            }}
        }}
        footer {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 12px 20px 32px;
            font-size: 0.8rem;
            color: #6b7280;
            border-top: 1px solid #e5e7eb;
        }}
    </style>
</head>
<body>
    <header>
        <h1>Calgary 311 Requests: Chart Dashboard</h1>
        <p>Input file: {html.escape(str(input_file))}</p>
        <p>Rows analyzed: {data_rows:,}</p>
    </header>
    <main>
        {highlights_card}
        {''.join(section_html)}
    </main>
    <script>
        const TOP_N = {top_n};
        const years = {years_json};
        const countsByYear = {counts_json};

        const startSelect = document.getElementById("start-year");
        const endSelect = document.getElementById("end-year");
        const imageEl = document.getElementById("top-services-image");
        const canvasWrap = document.getElementById("top-services-canvas-wrap");
        const canvasEl = document.getElementById("top-services-canvas");
        const rangeEl = document.getElementById("top-services-range");

        function formatNumber(n) {{
            return new Intl.NumberFormat().format(n);
        }}

        function populateYearSelect(selectEl) {{
            const allOpt = document.createElement("option");
            allOpt.value = "";
            allOpt.textContent = "All years";
            selectEl.appendChild(allOpt);

            years.forEach((y) => {{
                const opt = document.createElement("option");
                opt.value = String(y);
                opt.textContent = String(y);
                selectEl.appendChild(opt);
            }});
        }}

        function selectedYears() {{
            const start = startSelect.value ? parseInt(startSelect.value, 10) : null;
            const end = endSelect.value ? parseInt(endSelect.value, 10) : null;

            if (start !== null && end !== null && start > end) {{
                return [];
            }}

            return years.filter((y) => (start === null || y >= start) && (end === null || y <= end));
        }}

        function aggregateCounts(activeYears) {{
            const total = new Map();
            activeYears.forEach((y) => {{
                const yearly = countsByYear[String(y)] || {{}};
                Object.entries(yearly).forEach(([svc, count]) => {{
                    total.set(svc, (total.get(svc) || 0) + count);
                }});
            }});
            return [...total.entries()].sort((a, b) => b[1] - a[1]).slice(0, TOP_N);
        }}

        function renderCanvasChart(topServices) {{
            if (!canvasEl) {{
                return;
            }}

            const barCount = Math.max(topServices.length, 1);
            const rowHeight = 34;
            const leftPad = 310;
            const rightPad = 90;
            const topPad = 26;
            const width = 1100;
            const height = topPad + barCount * rowHeight + 20;

            canvasEl.width = width;
            canvasEl.height = height;

            const ctx = canvasEl.getContext("2d");
            if (!ctx) {{
                return;
            }}

            ctx.fillStyle = "#ffffff";
            ctx.fillRect(0, 0, width, height);

            if (topServices.length === 0) {{
                ctx.fillStyle = "#334155";
                ctx.font = "16px sans-serif";
                ctx.fillText("No services found for selected years.", 24, 42);
                return;
            }}

            const maxValue = topServices[0][1];
            const chartWidth = width - leftPad - rightPad;

            topServices.forEach(([service, count], idx) => {{
                const y = topPad + idx * rowHeight;
                const barWidth = maxValue > 0 ? (count / maxValue) * chartWidth : 0;

                ctx.fillStyle = "#1f2937";
                ctx.font = "13px sans-serif";
                ctx.textAlign = "right";
                ctx.textBaseline = "middle";
                const label = String(service).length > 45 ? String(service).slice(0, 42) + "..." : String(service);
                ctx.fillText(label, leftPad - 12, y + 10);

                ctx.fillStyle = "#0b5cab";
                ctx.fillRect(leftPad, y, barWidth, 20);

                ctx.fillStyle = "#334155";
                ctx.textAlign = "left";
                ctx.fillText(formatNumber(count), leftPad + barWidth + 8, y + 10);
            }});
        }}

        function updateTopServices() {{
            const activeYears = selectedYears();
            const topServices = aggregateCounts(activeYears);

            if (activeYears.length === 0) {{
                rangeEl.textContent = "Invalid range: start year is after end year.";
                if (canvasWrap) canvasWrap.style.display = "none";
                if (imageEl) imageEl.style.display = "block";
                return;
            }}

            const allYearsSelected = activeYears.length === years.length;
            rangeEl.textContent = allYearsSelected
                ? "Showing all years"
                : "Showing " + activeYears[0] + " to " + activeYears[activeYears.length - 1];

            if (allYearsSelected) {{
                if (canvasWrap) canvasWrap.style.display = "none";
                if (imageEl) imageEl.style.display = "block";
                return;
            }}

            if (imageEl) imageEl.style.display = "none";
            if (canvasWrap) canvasWrap.style.display = "block";
            renderCanvasChart(topServices);
        }}

        if (startSelect && endSelect && rangeEl) {{
            populateYearSelect(startSelect);
            populateYearSelect(endSelect);
            startSelect.addEventListener("change", updateTopServices);
            endSelect.addEventListener("change", updateTopServices);
            updateTopServices();
        }}
    </script>
    <footer>
        <p>* <strong>p95 resolution</strong>: the number of days within which 95% of service requests were closed. The remaining 5% took longer (outliers such as complex cases or requests left open for extended periods).</p>
    </footer>
</body>
</html>
"""

    html_path.write_text(html_doc, encoding="utf-8")
    return html_path


def main() -> None:
    args = parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    html_output = Path(args.html_output)
    html_output.parent.mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path, args.nrows)

    chart_sections = [
        {
            "id": "monthly_volume",
            "title": "Monthly 311 Request Volume",
            "summary": "Shows overall request demand over time to identify spikes and seasonality.",
            "image": None,
            "metrics": {},
        },
        {
            "id": "top_services",
            "title": f"Top {args.top_n} Service Request Types",
            "summary": "Interactive view: filter by request-year range using start/end dropdowns (or All years).",
            "image": None,
            "metrics": {},
        },
        {
            "id": "resolution_dist",
            "title": "Resolution Time Distribution",
            "summary": "Displays closure duration spread with key percentile markers for service-level performance.",
            "image": None,
            "metrics": {},
        },
        {
            "id": "open_closed",
            "title": "Open vs Closed by Request Month",
            "summary": "Dual-axis view with closed request volume and open-share percentage over time.",
            "image": None,
            "metrics": {},
        },
    ]

    outputs_and_metrics = [
        plot_monthly_requests(df, output_dir),
        plot_top_services(df, output_dir, args.top_n),
        plot_resolution_distribution(df, output_dir),
        plot_open_vs_closed_created_by_month(df, output_dir),
    ]

    for section, (image_path, metrics) in zip(chart_sections, outputs_and_metrics):
        section["image"] = image_path
        section["metrics"] = metrics

    service_years, service_counts_by_year = build_service_counts_by_year(df)

    monthly_metrics = outputs_and_metrics[0][1]
    service_metrics = outputs_and_metrics[1][1]
    resolution_metrics = outputs_and_metrics[2][1]
    highlights = {
        "Total rows analyzed": f"{len(df):,}",
        "Date range": monthly_metrics.get("Date range", ""),
        "Peak month": monthly_metrics.get("Peak month", ""),
        "Top service": service_metrics.get("Top service", ""),
        "Median resolution": resolution_metrics.get("Median (p50)", ""),
        "p95 resolution *": resolution_metrics.get("p95", ""),
    }

    html_path = render_html_dashboard(
        sections=chart_sections,
        html_path=html_output,
        data_rows=len(df),
        input_file=csv_path,
        top_n=args.top_n,
        service_years=service_years,
        service_counts_by_year=service_counts_by_year,
        highlights=highlights,
    )

    print("Created charts:")
    for out, _ in outputs_and_metrics:
        print(f"- {out}")
    print(f"Created HTML dashboard:\n- {html_path}")


if __name__ == "__main__":
    main()
