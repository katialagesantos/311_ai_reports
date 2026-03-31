"""
map_311_communities.py - Filterable interactive Calgary 311 map.
Data is sampled per year and embedded as JSON. A year-range picker
in the HTML lets the user select a range and render on demand.
Uses Leaflet.js (CDN) - no folium dependency.
"""
import argparse
import json
from pathlib import Path

import pandas as pd


CALGARY_CENTER = [51.0447, -114.0719]
CSV_PATH = "dataset/311_Service_Requests_20260329.csv"
DEFAULT_POINTS_PER_YEAR = 3_000


def parse_args():
    p = argparse.ArgumentParser(description="Generate a filterable interactive 311 map of Calgary.")
    p.add_argument("--input",  default=CSV_PATH)
    p.add_argument("--output", default="aea/reports/community_map.html")
    p.add_argument("--nrows",  type=int, default=None)
    p.add_argument("--points-per-year", type=int, default=DEFAULT_POINTS_PER_YEAR,
        help=f"Max sampled points per year (default: {DEFAULT_POINTS_PER_YEAR:,}).")
    return p.parse_args()


def load_and_aggregate(csv_path, nrows, points_per_year):
    cols = ["latitude", "longitude", "comm_name", "service_name", "requested_date"]
    print(f"Loading data from {csv_path} ...")
    df = pd.read_csv(csv_path, usecols=cols, low_memory=False, nrows=nrows)
    total = len(df)
    df = df.dropna(subset=["latitude", "longitude"])
    df["year"] = pd.to_datetime(df["requested_date"], errors="coerce").dt.year
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    kept = len(df)
    print(f"  Rows total: {total:,}  |  With coords+year: {kept:,}  |  Dropped: {total-kept:,}")

    df["latitude"]     = df["latitude"].round(4)
    df["longitude"]    = df["longitude"].round(4)
    df["service_name"] = df["service_name"].fillna("Unknown").str.slice(0, 40)
    df["comm_name"]    = df["comm_name"].fillna("Unknown").str.slice(0, 30)

    years = sorted(df["year"].unique().tolist())
    data_by_year = {}
    for year in years:
        chunk = df[df["year"] == year]
        if len(chunk) > points_per_year:
            chunk = chunk.sample(points_per_year, random_state=42)
        data_by_year[str(year)] = [
            [r.latitude, r.longitude, r.service_name, r.comm_name]
            for r in chunk.itertuples(index=False)
        ]
        print(f"  {year}: {len(data_by_year[str(year)]):,} points")
    return data_by_year, years


def build_html(data_by_year, years):
    min_year = min(years)
    max_year = max(years)
    year_opts = "\n".join(f'      <option value="{y}">{y}</option>' for y in years)
    data_json = json.dumps(data_by_year, separators=(",", ":"))
    lat, lon = CALGARY_CENTER

    # JS/CSS braces that are NOT f-string placeholders are doubled: {{ }}
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Calgary 311 Service Requests Map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css"/>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:Arial,sans-serif; display:flex; flex-direction:column; height:100vh; }}
  #controls {{
    background:#1e2a38; color:#eee; padding:10px 16px;
    display:flex; align-items:center; flex-wrap:wrap; gap:14px; z-index:1000;
  }}
  #controls h1 {{ font-size:15px; font-weight:bold; color:#fff; white-space:nowrap; }}
  .cg {{ display:flex; align-items:center; gap:6px; }}
  .cg label {{ font-size:12px; color:#aaa; }}
  .cg select {{
    background:#2c3e50; color:#fff; border:1px solid #4a6278;
    padding:4px 8px; border-radius:4px; font-size:13px; cursor:pointer;
  }}
  #btn {{
    background:#2980b9; color:white; border:none; padding:7px 18px;
    border-radius:4px; font-size:13px; cursor:pointer; font-weight:bold;
  }}
  #btn:hover {{ background:#3498db; }}
  #stats {{
    font-size:12px; color:#8ab; background:#16212d;
    padding:4px 12px; border-radius:4px;
  }}
  #map {{ flex:1; }}
  #legend {{
    position:absolute; bottom:28px; right:10px; z-index:999;
    background:rgba(255,255,255,0.95); padding:10px 14px;
    border-radius:6px; box-shadow:0 2px 6px rgba(0,0,0,0.2);
    font-size:12px; min-width:170px;
  }}
  #legend h4 {{ margin-bottom:8px; font-size:13px; }}
  .grad {{
    height:10px; width:100%; border-radius:3px; margin:4px 0 2px;
    background:linear-gradient(to right,
      rgba(0,0,255,.5),rgba(0,255,0,.7),
      rgba(255,255,0,.9),rgba(255,128,0,1),rgba(255,0,0,1));
  }}
  .lbl {{ display:flex; justify-content:space-between; color:#555; }}
  .lrow {{ display:flex; align-items:center; gap:8px; margin-top:6px; }}
  .cdot {{
    width:22px; height:22px; border-radius:50%; background:#1a73e8;
    color:white; font-size:10px; font-weight:bold;
    display:flex; align-items:center; justify-content:center; flex-shrink:0;
  }}
  .pdot {{
    width:12px; height:12px; border-radius:50%; background:#1a73e8;
    border:2px solid white; box-shadow:0 0 3px rgba(0,0,0,.4); flex-shrink:0;
  }}
</style>
</head>
<body>

<div id="controls">
  <h1>&#128506; Calgary 311 Service Requests</h1>
  <div class="cg">
    <label>From</label>
    <select id="ys">
{year_opts}
    </select>
  </div>
  <div class="cg">
    <label>To</label>
    <select id="ye">
{year_opts}
    </select>
  </div>
  <div class="cg">
    <label>Mode&nbsp;</label>
    <label><input type="radio" name="mode" value="heatmap" checked> Heatmap</label>&nbsp;
    <label><input type="radio" name="mode" value="cluster"> Clusters</label>&nbsp;
    <label><input type="radio" name="mode" value="both"> Both</label>
  </div>
  <button id="btn" onclick="go()">&#9654; Generate Map</button>
  <div id="stats">Select a year range and click Generate Map</div>
</div>

<div id="map"></div>

<div id="legend">
  <h4>Legend</h4>
  <div id="lh">
    <b style="font-size:11px;">Request density</b>
    <div class="grad"></div>
    <div class="lbl"><span>Low</span><span>High</span></div>
  </div>
  <div id="lc" style="display:none">
    <b style="font-size:11px;">Request locations</b>
    <div class="lrow">
      <div class="cdot">N</div>
      <span style="color:#555">Cluster (zoom to expand)</span>
    </div>
    <div class="lrow">
      <div class="pdot"></div>
      <span style="color:#555">Single request</span>
    </div>
  </div>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.heat/dist/leaflet-heat.js"></script>
<script src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>
<script>
const D = {data_json};
const MIN = {min_year}, MAX = {max_year};

const map = L.map('map').setView([{lat}, {lon}], 11);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  attribution: '&copy; OpenStreetMap &copy; CARTO', maxZoom: 19
}}).addTo(map);

let HL = null, CL = null;
const ys = document.getElementById('ys');
const ye = document.getElementById('ye');
ys.value = MIN;
ye.value = MAX;

function go() {{
  const s = +ys.value, e = +ye.value;
  const mode = document.querySelector('[name=mode]:checked').value;
  if (s > e) {{ alert('Start year must be \u2264 end year.'); return; }}

  const hp = [], cp = [], sc = {{}}, cc = {{}};
  for (let y = s; y <= e; y++) {{
    for (const [lt, ln, sv, cm] of (D[y] || [])) {{
      hp.push([lt, ln, 1]);
      cp.push({{ lt, ln, sv, cm }});
      sc[sv] = (sc[sv] || 0) + 1;
      cc[cm] = (cc[cm] || 0) + 1;
    }}
  }}

  if (HL) {{ map.removeLayer(HL); HL = null; }}
  if (CL) {{ map.removeLayer(CL); CL = null; }}

  if (mode === 'heatmap' || mode === 'both')
    HL = L.heatLayer(hp, {{ radius: 12, blur: 15, minOpacity: 0.3 }}).addTo(map);

  if (mode === 'cluster' || mode === 'both') {{
    CL = L.markerClusterGroup({{ disableClusteringAtZoom: 15, chunkedLoading: true }});
    for (const p of cp)
      L.circleMarker([p.lt, p.ln], {{
        radius: 5, color: '#1a73e8', fillColor: '#1a73e8', fillOpacity: 0.7, weight: 1
      }}).bindPopup('<b>' + p.sv + '</b><br>' + p.cm).addTo(CL);
    CL.addTo(map);
  }}

  const ts = Object.entries(sc).sort((a, b) => b[1] - a[1])[0];
  const tc = Object.entries(cc).sort((a, b) => b[1] - a[1])[0];
  document.getElementById('stats').innerHTML =
    '<b>' + hp.length.toLocaleString() + '</b> points &nbsp;|&nbsp; ' +
    'Top service: <b>' + (ts ? ts[0] : 'N/A') + '</b> &nbsp;|&nbsp; ' +
    'Top community: <b>' + (tc ? tc[0] : 'N/A') + '</b>';

  document.getElementById('lh').style.display = (mode !== 'cluster') ? 'block' : 'none';
  document.getElementById('lc').style.display = (mode !== 'heatmap') ? 'block' : 'none';
}}

go();
</script>
</body>
</html>"""


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data_by_year, years = load_and_aggregate(args.input, args.nrows, args.points_per_year)

    print("Building HTML map ...")
    html = build_html(data_by_year, years)
    output_path.write_text(html, encoding="utf-8")

    size_mb = output_path.stat().st_size / 1_048_576
    print(f"\nMap saved to: {output_path}  ({size_mb:.1f} MB)")
    print(f"Years: {min(years)}-{max(years)}  |  Max points/year: {args.points_per_year:,}")


if __name__ == "__main__":
    main()
