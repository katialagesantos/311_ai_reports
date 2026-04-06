"""
weather_311_analysis.py
Discovers hidden relationships between Calgary weather conditions,
311 service request types, and communities.

Weather source : Open-Meteo historical archive API (free, no API key).
Output         : docs/weather_311_analysis.html  (self-contained)

Dependencies: pandas, matplotlib, numpy, requests
"""
import argparse
import base64
import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests

# ── config ────────────────────────────────────────────────────────────────────
CALGARY_LAT   = 51.0447
CALGARY_LON   = -114.0719
WEATHER_CACHE = Path("aea/output/weather_calgary_daily.csv")
CSV_PATH      = "dataset/311_Service_Requests_20260329.csv"
DEFAULT_OUT   = "docs/weather_311_analysis.html"

WEATHER_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "snowfall_sum",
    "wind_speed_10m_max",
]
WEATHER_LABELS = {
    "temperature_2m_max": "Max Temp (°C)",
    "temperature_2m_min": "Min Temp (°C)",
    "precipitation_sum":  "Precipitation (mm)",
    "snowfall_sum":       "Snowfall (cm)",
    "wind_speed_10m_max": "Max Wind (km/h)",
}

# Primary weather category — first matching rule wins (priority order)
CAT_RULES = [
    ("Extreme Cold", lambda d: d["temperature_2m_max"] < -15),
    ("Heavy Snow",   lambda d: d["snowfall_sum"] >= 10),
    ("Heavy Rain",   lambda d: d["precipitation_sum"] >= 20),
    ("Windy",        lambda d: d["wind_speed_10m_max"] >= 55),
    ("Hot",          lambda d: d["temperature_2m_max"] >= 28),
    ("Cold",         lambda d: d["temperature_2m_max"] < 0),
    ("Snow",         lambda d: d["snowfall_sum"] >= 2),
    ("Rain",         lambda d: d["precipitation_sum"] >= 5),
    ("Cool",         lambda d: d["temperature_2m_max"] < 15),
    ("Warm/Normal",  lambda d: pd.Series([True] * len(d), index=d.index)),
]

SEASONS = {
    1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring",  5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall",   10: "Fall",
    11: "Fall",  12: "Winter",
}


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Analyse relationships between Calgary 311 requests and weather."
    )
    p.add_argument("--input",        default=CSV_PATH)
    p.add_argument("--output",       default=DEFAULT_OUT)
    p.add_argument("--nrows",        type=int, default=None,
        help="Limit rows loaded from CSV (useful for quick tests).")
    p.add_argument("--top-services", type=int, default=20,
        help="Number of top services shown in correlation/seasonal charts.")
    p.add_argument("--no-cache",     action="store_true",
        help="Re-fetch weather data even if local cache exists.")
    return p.parse_args()


# ── weather ───────────────────────────────────────────────────────────────────
def fetch_weather(start_date: str, end_date: str, no_cache: bool = False) -> pd.DataFrame:
    """Fetch daily Calgary weather from Open-Meteo archive, with local CSV cache."""
    if WEATHER_CACHE.exists() and not no_cache:
        print(f"  Loading cached weather from {WEATHER_CACHE}")
        return pd.read_csv(WEATHER_CACHE, parse_dates=["date"])

    vars_str = ",".join(WEATHER_VARS)
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={CALGARY_LAT}&longitude={CALGARY_LON}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily={vars_str}"
        f"&timezone=America%2FEdmonton"
    )
    print(f"  Fetching weather from Open-Meteo ({start_date} → {end_date}) ...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()["daily"]

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])

    WEATHER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(WEATHER_CACHE, index=False)
    print(f"  Weather cached → {WEATHER_CACHE}  ({len(df):,} days)")
    return df


def assign_weather_category(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a single primary weather category and season to each day."""
    cats = pd.Series("Warm/Normal", index=df.index)
    for name, rule in reversed(CAT_RULES[:-1]):
        cats[rule(df)] = name
    df = df.copy()
    df["weather_cat"] = cats
    df["month"]  = df["date"].dt.month
    df["season"] = df["month"].map(SEASONS)
    return df


# ── 311 data ──────────────────────────────────────────────────────────────────
def load_311(csv_path: str, nrows) -> pd.DataFrame:
    cols = ["requested_date", "service_name", "comm_name"]
    print(f"  Loading 311 data from {csv_path} ...")
    df = pd.read_csv(csv_path, usecols=cols, low_memory=False, nrows=nrows)
    df["date"] = pd.to_datetime(df["requested_date"], errors="coerce", format="mixed")
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.normalize()
    df["service_name"] = df["service_name"].fillna("Unknown").str.strip()
    df["comm_name"]    = df["comm_name"].fillna("Unknown").str.strip()
    print(
        f"  {len(df):,} requests  |  "
        f"{df['date'].min().date()} – {df['date'].max().date()}"
    )
    return df


def merge_data(requests_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    merged = requests_df.merge(weather_df, on="date", how="inner")
    print(f"  Merged: {len(merged):,} requests with weather data")
    return merged


# ── chart helpers ─────────────────────────────────────────────────────────────
def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ── analyses ──────────────────────────────────────────────────────────────────
def plot_correlation_heatmap(merged: pd.DataFrame, top_n: int) -> str:
    """Spearman correlation: daily service counts vs weather variables."""
    daily_counts = (
        merged.groupby(["date", "service_name"])
        .size()
        .unstack(fill_value=0)
    )
    daily_weather = (
        merged.drop_duplicates("date")
        .set_index("date")[WEATHER_VARS]
    )
    combined = daily_counts.join(daily_weather, how="inner")

    top_services = [
        s for s in merged["service_name"].value_counts().head(top_n).index
        if s in daily_counts.columns
    ]

    corr_matrix = pd.DataFrame({
        label: combined[top_services].corrwith(combined[var], method="spearman")
        for var, label in WEATHER_LABELS.items()
    }).reindex(top_services)

    corr_matrix["_sort"] = corr_matrix.abs().max(axis=1)
    corr_matrix = corr_matrix.sort_values("_sort", ascending=False).drop(columns="_sort")

    fig, ax = plt.subplots(figsize=(9, max(6, len(corr_matrix) * 0.38)))
    im = ax.imshow(corr_matrix.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(corr_matrix)))
    ax.set_yticklabels(corr_matrix.index, fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.6, label="Spearman ρ")

    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix.columns)):
            val = corr_matrix.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(val) > 0.4 else "black")

    ax.set_title(
        f"Spearman Correlation: Daily Service Counts vs Weather Variables\n"
        f"(top {top_n} services by volume — sorted by max |ρ|)",
        fontsize=11
    )
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_seasonal_heatmap(merged: pd.DataFrame, top_n: int) -> str:
    """Row-normalised heatmap: service_name × month."""
    merged = merged.copy()
    merged["month"] = merged["date"].dt.month
    top_services = merged["service_name"].value_counts().head(top_n).index

    pivot = (
        merged[merged["service_name"].isin(top_services)]
        .groupby(["service_name", "month"])
        .size()
        .unstack(fill_value=0)
    )
    pivot_norm = pivot.div(pivot.sum(axis=1), axis=0)
    pivot_norm = pivot_norm.loc[pivot_norm.idxmax(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(11, max(6, len(pivot_norm) * 0.38)))
    im = ax.imshow(pivot_norm.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(12))
    ax.set_xticklabels(
        ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
        fontsize=9
    )
    ax.set_yticks(range(len(pivot_norm)))
    ax.set_yticklabels(pivot_norm.index, fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.6, label="Share of annual requests")
    ax.set_title(
        f"Seasonal Distribution of Service Requests\n"
        f"(normalised row share — top {top_n} services, sorted by peak month)",
        fontsize=11
    )
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_weather_category_breakdown(merged: pd.DataFrame, top_n: int) -> str:
    """Average daily requests per service type by weather category."""
    top_services  = merged["service_name"].value_counts().head(top_n).index
    days_per_cat  = merged.drop_duplicates("date")["weather_cat"].value_counts()

    sub = merged[merged["service_name"].isin(top_services)]
    pivot = (
        sub.groupby(["weather_cat", "service_name"])
        .size()
        .unstack(fill_value=0)
        .astype(float)
    )
    for cat in pivot.index:
        n = days_per_cat.get(cat, 1)
        pivot.loc[cat] = pivot.loc[cat] / max(n, 1)

    cat_order = [
        "Extreme Cold", "Cold", "Cool", "Warm/Normal", "Hot",
        "Snow", "Heavy Snow", "Rain", "Heavy Rain", "Windy",
    ]
    pivot = pivot.reindex([c for c in cat_order if c in pivot.index])

    fig, ax = plt.subplots(figsize=(13, 6))
    pivot.plot(kind="bar", ax=ax, width=0.8, colormap="tab20")
    ax.set_xlabel("Weather Category")
    ax.set_ylabel("Avg Daily Requests")
    ax.set_title(
        f"Average Daily Requests by Weather Category\n(top {top_n} service types)"
    )
    ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.7)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig_to_b64(fig)


def _community_extreme_chart(data: pd.DataFrame, extreme_cats: list, top_n: int, title: str) -> str:
    if data.empty:
        return None
    comm_cat = (
        data.groupby(["comm_name", "weather_cat"])
        .size()
        .unstack(fill_value=0)
    )
    comm_cat["total"] = comm_cat.sum(axis=1)
    top_comms = comm_cat.nlargest(top_n, "total").drop(columns="total")
    present   = [c for c in extreme_cats if c in top_comms.columns]

    fig, ax = plt.subplots(figsize=(11, 6))
    top_comms[present].plot(kind="barh", stacked=True, ax=ax, colormap="Set2")
    ax.set_xlabel("Number of Requests")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_community_extreme_weather(merged: pd.DataFrame, top_n: int = 15) -> tuple:
    """Returns two charts: one including unknown communities, one excluding them."""
    extreme_cats = ["Extreme Cold", "Heavy Snow", "Heavy Rain", "Windy"]
    extreme_all = merged[merged["weather_cat"].isin(extreme_cats)]
    extreme_known = extreme_all[extreme_all["comm_name"] != "Unknown"]

    chart_all   = _community_extreme_chart(
        extreme_all, extreme_cats, top_n,
        f"Top {top_n} Communities During Extreme Weather (all requests)"
    )
    chart_known = _community_extreme_chart(
        extreme_known, extreme_cats, top_n,
        f"Top {top_n} Communities During Extreme Weather (assigned communities only)"
    )
    return chart_all, chart_known


def plot_monthly_volume_temp(merged: pd.DataFrame) -> str:
    """Monthly 311 volume (bars) overlaid with average max temperature (line)."""
    merged = merged.copy()
    merged["ym"] = merged["date"].dt.to_period("M")

    monthly_counts = merged.groupby("ym").size()
    monthly_temp   = (
        merged.drop_duplicates("date")
        .assign(ym=lambda d: d["date"].dt.to_period("M"))
        .groupby("ym")["temperature_2m_max"]
        .mean()
    )

    idx = monthly_counts.index.union(monthly_temp.index)
    monthly_counts = monthly_counts.reindex(idx, fill_value=0)
    monthly_temp   = monthly_temp.reindex(idx)

    x      = range(len(idx))
    labels = [str(p) for p in idx]
    step   = max(1, len(labels) // 20)

    fig, ax1 = plt.subplots(figsize=(14, 4))
    ax2 = ax1.twinx()

    ax1.bar(x, monthly_counts.values, color="#2980b9", alpha=0.6, label="311 Requests")
    ax2.plot(x, monthly_temp.values, color="#e74c3c", linewidth=1.5, label="Max Temp (°C)")
    ax2.axhline(0, color="#e74c3c", linewidth=0.5, linestyle="--", alpha=0.4)

    ax1.set_xticks([i for i in x if i % step == 0])
    ax1.set_xticklabels([labels[i] for i in x if i % step == 0],
                        rotation=45, ha="right", fontsize=7)
    ax1.set_ylabel("Monthly Request Count", color="#2980b9")
    ax2.set_ylabel("Avg Max Temp (°C)",     color="#e74c3c")
    ax1.set_title("Monthly 311 Request Volume vs Average Max Temperature")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig_to_b64(fig)


def get_spike_results(merged: pd.DataFrame, top_n: int = 10) -> dict:
    """For each extreme weather category, find services with highest spike ratio vs normal days."""
    normal = merged[merged["weather_cat"] == "Warm/Normal"]
    n_normal_days = normal["date"].nunique()
    if n_normal_days == 0:
        return {}
    normal_daily = normal.groupby("service_name").size() / n_normal_days

    top_services = set(merged["service_name"].value_counts().head(100).index)
    results = {}
    for cat in ["Extreme Cold", "Heavy Snow", "Heavy Rain", "Windy", "Hot"]:
        sub = merged[merged["weather_cat"] == cat]
        n_days = sub["date"].nunique()
        if len(sub) < 50 or n_days == 0:
            continue
        cat_daily = sub.groupby("service_name").size() / n_days
        ratio = (
            (cat_daily / normal_daily)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        ratio = ratio[ratio.index.isin(top_services)]
        results[cat] = ratio.sort_values(ascending=False).head(top_n)
    return results


# ── report ────────────────────────────────────────────────────────────────────
def build_report(imgs: dict, spike_results: dict, merged: pd.DataFrame) -> str:
    n_requests    = len(merged)
    date_range    = f"{merged['date'].min().date()} – {merged['date'].max().date()}"
    n_services    = merged["service_name"].nunique()
    n_communities = merged["comm_name"].nunique()
    n_weather_days = merged["date"].nunique()

    def img_tag(key):
        v = imgs.get(key)
        return (
            f'<img src="data:image/png;base64,{v}" style="max-width:100%;border-radius:6px;">'
            if v else "<p style='color:#999'>No data available.</p>"
        )

    # Spike tables
    spike_html = ""
    for cat, sr in spike_results.items():
        rows = "".join(
            f"<tr><td>{svc}</td><td style='text-align:right'>{ratio:.2f}×</td></tr>"
            for svc, ratio in sr.items()
        )
        spike_html += f"""
        <div class="spike-block">
          <h4>{cat}</h4>
          <table class="spike-table">
            <tr><th>Service Type</th><th>Spike</th></tr>
            {rows}
          </table>
        </div>"""

    # Weather category day counts
    cat_counts = (
        merged.drop_duplicates("date")["weather_cat"]
        .value_counts()
        .to_dict()
    )
    cat_rows = "".join(
        f"<tr><td>{cat}</td><td style='text-align:right'>{cnt:,}</td></tr>"
        for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1])
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Calgary 311 – Weather Relationship Analysis</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: Arial, sans-serif; background: #f4f6f9; color: #222; }}
  header {{ background: #1e2a38; color: white; padding: 20px 32px; }}
  header h1 {{ font-size: 22px; margin-bottom: 4px; }}
  header p  {{ color: #8ab; font-size: 13px; }}
  .stats-bar {{
    display: flex; gap: 14px; flex-wrap: wrap;
    background: #16212d; padding: 12px 32px;
  }}
  .stat {{
    background: #2c3e50; border-radius: 6px;
    padding: 10px 20px; text-align: center;
  }}
  .stat .val {{ font-size: 22px; font-weight: bold; color: #3498db; }}
  .stat .lbl {{ font-size: 11px; color: #aaa; margin-top: 2px; }}
  .section {{
    margin: 24px 32px; background: white;
    border-radius: 8px; padding: 20px 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,.08);
  }}
  .section h2 {{ font-size: 16px; color: #2c3e50; margin-bottom: 6px; }}
  .section p.desc {{ font-size: 13px; color: #666; margin-bottom: 16px; }}
  .two-col {{ display: flex; gap: 24px; flex-wrap: wrap; }}
  .two-col > * {{ flex: 1; min-width: 260px; }}
  .spike-grid {{ display: flex; flex-wrap: wrap; gap: 24px; margin-top: 14px; }}
  .spike-block h4 {{ font-size: 13px; color: #2980b9; margin-bottom: 6px; }}
  .spike-table {{ border-collapse: collapse; font-size: 12px; min-width: 230px; width: 100%; }}
  .spike-table th {{
    background: #2c3e50; color: white;
    padding: 5px 10px; text-align: left;
  }}
  .spike-table td {{ padding: 4px 10px; border-bottom: 1px solid #eee; }}
  .spike-table tr:nth-child(even) {{ background: #f8f9fa; }}
  .cat-table {{ border-collapse: collapse; font-size: 12px; width: 100%; max-width: 320px; }}
  .cat-table th {{ background: #2c3e50; color: white; padding: 5px 12px; text-align: left; }}
  .cat-table td {{ padding: 4px 12px; border-bottom: 1px solid #eee; }}
  .cat-table tr:nth-child(even) {{ background: #f8f9fa; }}
  footer {{
    text-align: center; padding: 20px;
    color: #999; font-size: 12px; margin-top: 10px;
  }}
</style>
</head>
<body>

<header>
  <h1>&#127774; Calgary 311 – Weather Relationship Analysis</h1>
  <p>Discovering hidden correlations between weather conditions, service request types, and communities</p>
</header>

<div class="stats-bar">
  <div class="stat"><div class="val">{n_requests:,}</div><div class="lbl">Requests Analysed</div></div>
  <div class="stat"><div class="val">{n_weather_days:,}</div><div class="lbl">Days With Weather</div></div>
  <div class="stat"><div class="val">{n_services}</div><div class="lbl">Service Types</div></div>
  <div class="stat"><div class="val">{n_communities}</div><div class="lbl">Communities</div></div>
  <div class="stat"><div class="val" style="font-size:14px">{date_range}</div><div class="lbl">Date Range</div></div>
</div>

<div class="section">
  <h2>Weather Day Breakdown</h2>
  <p class="desc">Number of days in the dataset classified into each weather category (primary condition only).</p>
  <table class="cat-table">
    <tr><th>Weather Category</th><th>Days</th></tr>
    {cat_rows}
  </table>
</div>

<div class="section">
  <h2>1. Correlation Heatmap: Service Types vs Weather Variables</h2>
  <p class="desc">
    Spearman correlation between daily request counts per service type and daily weather values.
    <strong>Red</strong> = more requests on days with higher values of that variable.
    <strong>Blue</strong> = fewer requests. Sorted by highest absolute correlation.
  </p>
  {img_tag("corr")}
</div>

<div class="section">
  <h2>2. Seasonal Distribution by Service Type</h2>
  <p class="desc">
    Each row shows what share of that service's annual requests fall in each month.
    Services are sorted by their peak month — revealing clear seasonal clusters.
  </p>
  {img_tag("seasonal")}
</div>

<div class="section">
  <h2>3. Average Daily Requests by Weather Category</h2>
  <p class="desc">
    Average number of requests per service type on days classified into each weather category.
    Normalised by the number of days in each category so categories with fewer days are comparable.
  </p>
  {img_tag("weather_cat")}
</div>

<div class="section">
  <h2>4. Monthly Volume vs Temperature</h2>
  <p class="desc">
    Total 311 requests per month (blue bars) vs average daily maximum temperature (red line).
    Reveals how seasonal temperature changes drive overall request volume.
  </p>
  {img_tag("monthly")}
</div>

<div class="section">
  <h2>5a. Communities Most Affected During Extreme Weather (all requests)</h2>
  <p class="desc">
    Top communities by total request volume on extreme weather days, including requests with no assigned community (grouped as "Unknown").
    Stacked bars show which weather type drove each community's requests.
  </p>
  {img_tag("community_all")}
  <p style="font-size:11px;color:#999;margin-top:10px;">
    * <strong>Unknown</strong> represents service requests where the <code>comm_name</code> field is empty (no community was recorded at time of submission).
  </p>
</div>

<div class="section">
  <h2>5b. Communities Most Affected During Extreme Weather (assigned communities only)</h2>
  <p class="desc">
    Same chart as above but excluding requests with no community assigned, showing only requests tied to a specific Calgary community.
  </p>
  {img_tag("community_known")}
</div>

<div class="section">
  <h2>6. Service Type Spikes During Extreme Weather</h2>
  <p class="desc">
    Ratio of average daily requests during each weather event vs Warm/Normal days.
    A value of <strong>2.5×</strong> means 2.5× more requests on average than a normal day.
    Only services in the top 100 by volume are included.
  </p>
  <div class="spike-grid">{spike_html}</div>
</div>

<footer>
  Data: City of Calgary 311 Open Data &amp; Open-Meteo Historical Archive &nbsp;|&nbsp;
  Generated {pd.Timestamp.now().strftime("%Y-%m-%d")}
</footer>
</body>
</html>"""


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Loading 311 data ...")
    requests_df = load_311(args.input, args.nrows)
    start_date  = requests_df["date"].min().strftime("%Y-%m-%d")
    end_date    = requests_df["date"].max().strftime("%Y-%m-%d")

    print("\n[2/5] Fetching weather data ...")
    weather_df = fetch_weather(start_date, end_date, no_cache=args.no_cache)
    weather_df = assign_weather_category(weather_df)

    print("\n[3/5] Merging datasets ...")
    merged = merge_data(requests_df, weather_df)

    print("\n[4/5] Running analyses ...")
    community_all, community_known = plot_community_extreme_weather(merged)
    imgs = {
        "corr":           plot_correlation_heatmap(merged, args.top_services),
        "seasonal":       plot_seasonal_heatmap(merged, args.top_services),
        "weather_cat":    plot_weather_category_breakdown(merged, min(15, args.top_services)),
        "community_all":  community_all,
        "community_known": community_known,
        "monthly":        plot_monthly_volume_temp(merged),
    }
    spike_results = get_spike_results(merged)

    print("\n[5/5] Building HTML report ...")
    html = build_report(imgs, spike_results, merged)
    output_path.write_text(html, encoding="utf-8")

    size_mb = output_path.stat().st_size / 1_048_576
    print(f"\nReport saved to: {output_path}  ({size_mb:.1f} MB)")
    print(f"Weather categories found: {sorted(merged['weather_cat'].unique())}")


if __name__ == "__main__":
    main()
