import argparse
import sys
from pathlib import Path

import pandas as pd
import folium
from folium.plugins import HeatMap, FastMarkerCluster


CALGARY_CENTER = [51.0447, -114.0719]
CSV_PATH = "dataset/311_Service_Requests_20260329.csv"
HEATMAP_MAX_POINTS = 200_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an interactive 311 request map of Calgary using lat/lon data."
    )
    parser.add_argument(
        "--input",
        default=CSV_PATH,
        help=f"Path to the 311 CSV file (default: {CSV_PATH}).",
    )
    parser.add_argument(
        "--output",
        default="docs/community_map.html",
        help="Output path for the HTML map (default: docs/community_map.html).",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Optional row limit for faster runs (default: all rows).",
    )
    parser.add_argument(
        "--mode",
        choices=["heatmap", "cluster", "both"],
        default="both",
        help="Map mode: heatmap, cluster, or both (default: both).",
    )
    parser.add_argument(
        "--heatmap-sample",
        type=int,
        default=HEATMAP_MAX_POINTS,
        help=f"Max points used for the heatmap layer (default: {HEATMAP_MAX_POINTS:,}).",
    )
    return parser.parse_args()


def load_data(csv_path: str, nrows: int | None) -> pd.DataFrame:
    cols = ["latitude", "longitude", "comm_name", "service_name"]
    print(f"Loading data from {csv_path} ...")
    df = pd.read_csv(csv_path, usecols=cols, low_memory=False, nrows=nrows)
    total = len(df)
    df = df.dropna(subset=["latitude", "longitude"])
    kept = len(df)
    print(f"  Rows loaded: {total:,}  |  With coordinates: {kept:,}  |  Dropped: {total - kept:,}")
    return df


def build_map(df: pd.DataFrame, mode: str, heatmap_sample: int) -> folium.Map:
    m = folium.Map(
        location=CALGARY_CENTER,
        zoom_start=11,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # ── Heatmap layer ──────────────────────────────────────────────────────────
    if mode in ("heatmap", "both"):
        sample = df if len(df) <= heatmap_sample else df.sample(heatmap_sample, random_state=42)
        heat_data = sample[["latitude", "longitude"]].values.tolist()
        HeatMap(
            heat_data,
            name="Request density (heatmap)",
            min_opacity=0.3,
            radius=10,
            blur=12,
        ).add_to(m)
        print(f"  Heatmap: {len(heat_data):,} points")

    # ── Cluster layer ──────────────────────────────────────────────────────────
    if mode in ("cluster", "both"):
        points = df[["latitude", "longitude"]].values.tolist()
        FastMarkerCluster(
            points,
            name="Request locations (clusters)",
            disableClusteringAtZoom=15,
        ).add_to(m)
        print(f"  Cluster: {len(points):,} points")

    folium.LayerControl(collapsed=False).add_to(m)

    # ── Stats popup in title card ──────────────────────────────────────────────
    total_requests = len(df)
    top_community = (
        df["comm_name"].value_counts().idxmax()
        if "comm_name" in df.columns and df["comm_name"].notna().any()
        else "N/A"
    )
    top_service = (
        df["service_name"].value_counts().idxmax()
        if "service_name" in df.columns and df["service_name"].notna().any()
        else "N/A"
    )

    title_html = f"""
    <div style="
        position: fixed;
        top: 12px; left: 55px; z-index: 1000;
        background: white;
        padding: 10px 16px;
        border-radius: 6px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.25);
        font-family: Arial, sans-serif;
        font-size: 13px;
        max-width: 280px;
    ">
        <strong style="font-size:15px;">Calgary 311 Service Requests</strong><br>
        <span style="color:#555;">Requests plotted: <b>{total_requests:,}</b></span><br>
        <span style="color:#555;">Top community: <b>{top_community}</b></span><br>
        <span style="color:#555;">Top service: <b>{top_service}</b></span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px; right: 12px; z-index: 1000;
        background: white;
        padding: 12px 16px;
        border-radius: 6px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.25);
        font-family: Arial, sans-serif;
        font-size: 12px;
        min-width: 180px;
    ">
        <strong style="font-size:13px; display:block; margin-bottom:8px;">Legend</strong>

        <div style="margin-bottom:6px;">
            <span style="font-weight:bold; display:block; margin-bottom:4px;">Heatmap — request density</span>
            <div style="
                height: 12px;
                width: 100%;
                border-radius: 3px;
                background: linear-gradient(to right,
                    rgba(0,0,255,0.4),
                    rgba(0,255,0,0.6),
                    rgba(255,255,0,0.8),
                    rgba(255,128,0,0.9),
                    rgba(255,0,0,1));
            "></div>
            <div style="display:flex; justify-content:space-between; color:#555; margin-top:2px;">
                <span>Low</span><span>High</span>
            </div>
        </div>

        <div style="margin-top:10px;">
            <span style="font-weight:bold; display:block; margin-bottom:4px;">Clusters — individual requests</span>
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:3px;">
                <div style="
                    width:26px; height:26px; border-radius:50%;
                    background:#1a73e8; color:white;
                    display:flex; align-items:center; justify-content:center;
                    font-size:11px; font-weight:bold; flex-shrink:0;">
                    N
                </div>
                <span style="color:#555;">Grouped cluster (zoom in to expand)</span>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
                <div style="
                    width:14px; height:14px; border-radius:50%;
                    background:#1a73e8; border:2px solid white;
                    box-shadow:0 0 3px rgba(0,0,0,0.4); flex-shrink:0;">
                </div>
                <span style="color:#555;">Single request location</span>
            </div>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


def main() -> None:
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_data(args.input, args.nrows)

    print(f"Building map (mode={args.mode}) ...")
    m = build_map(df, mode=args.mode, heatmap_sample=args.heatmap_sample)

    m.save(str(output_path))
    print(f"\nMap saved to: {output_path}")


if __name__ == "__main__":
    main()
