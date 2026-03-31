import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_CATEGORICAL_COLUMNS = [
    "status_description",
    "service_name",
    "agency_responsible",
    "source",
    "location_type",
    "comm_name",
]

DEFAULT_DATE_COLUMNS = [
    "requested_date",
    "updated_date",
    "closed_date",
]


def series_to_counts(series: pd.Series, top_n: int) -> dict[str, int]:
    counts = series.astype("string").fillna("MISSING").value_counts().head(top_n)
    return {str(k): int(v) for k, v in counts.items()}


def describe_resolution_days(requested: pd.Series, closed: pd.Series) -> dict[str, float]:
    req = pd.to_datetime(requested, errors="coerce", format="mixed")
    cls = pd.to_datetime(closed, errors="coerce", format="mixed")
    days = (cls - req).dt.total_seconds() / 86400
    valid = days[(days >= 0) & days.notna()]

    if valid.empty:
        return {}

    desc = valid.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])
    return {k: round(float(v), 2) for k, v in desc.to_dict().items()}


def analyze(csv_path: Path, top_n: int, nrows: int | None) -> dict[str, Any]:
    df = pd.read_csv(csv_path, low_memory=False, nrows=nrows)

    result: dict[str, Any] = {
        "input_file": str(csv_path),
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": list(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    result["top_missing_pct"] = {
        str(col): round(float(pct), 2)
        for col, pct in missing_pct.head(top_n).items()
    }

    categorical_summary: dict[str, dict[str, int]] = {}
    for col in DEFAULT_CATEGORICAL_COLUMNS:
        if col in df.columns:
            categorical_summary[col] = series_to_counts(df[col], top_n)
    result["categorical_top_values"] = categorical_summary

    date_summary: dict[str, dict[str, Any]] = {}
    for col in DEFAULT_DATE_COLUMNS:
        if col in df.columns:
            d = pd.to_datetime(df[col], errors="coerce", format="mixed")
            date_summary[col] = {
                "missing_pct": round(float(d.isna().mean() * 100), 2),
                "min": str(d.min()) if not d.dropna().empty else None,
                "max": str(d.max()) if not d.dropna().empty else None,
            }
    result["date_summary"] = date_summary

    if {"requested_date", "closed_date"}.issubset(df.columns):
        result["resolution_days"] = describe_resolution_days(
            df["requested_date"], df["closed_date"]
        )

    if "requested_date" in df.columns:
        req = pd.to_datetime(df["requested_date"], errors="coerce", format="mixed")
        month_counts = req.dt.to_period("M").value_counts().sort_values(ascending=False)
        result["top_request_months"] = {
            str(month): int(count)
            for month, count in month_counts.head(top_n).items()
        }

    return result


def print_report(result: dict[str, Any]) -> None:
    shape = result["shape"]
    print(f"Rows: {shape['rows']:,}")
    print(f"Columns: {shape['columns']}")
    print(f"Duplicate rows: {result['duplicate_rows']:,}")

    print("\nTop missing columns (%):")
    for col, pct in result.get("top_missing_pct", {}).items():
        print(f"  - {col}: {pct}%")

    print("\nTop categorical values:")
    for col, values in result.get("categorical_top_values", {}).items():
        print(f"  - {col}")
        for key, count in values.items():
            print(f"      {key}: {count:,}")

    print("\nDate coverage:")
    for col, details in result.get("date_summary", {}).items():
        print(
            f"  - {col}: missing={details['missing_pct']}% "
            f"range=[{details['min']} .. {details['max']}]"
        )

    resolution = result.get("resolution_days", {})
    if resolution:
        print("\nResolution days stats:")
        keys = ["count", "mean", "50%", "75%", "90%", "95%", "99%", "max"]
        for k in keys:
            if k in resolution:
                print(f"  - {k}: {resolution[k]}")

    print("\nTop request months:")
    for month, count in result.get("top_request_months", {}).items():
        print(f"  - {month}: {count:,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Calgary 311 service request data."
    )
    parser.add_argument(
        "--input",
        default="dataset/311_Service_Requests_20260329.csv",
        help="Path to CSV input file.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top categories/months/missing columns to show.",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Optional row limit for quick testing.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save full summary as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.input)

    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    result = analyze(csv_path, top_n=args.top_n, nrows=args.nrows)
    print_report(result)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nSaved JSON summary to: {out_path}")


if __name__ == "__main__":
    main()
