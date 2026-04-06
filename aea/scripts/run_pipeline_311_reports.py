import argparse
import subprocess
import sys
from pathlib import Path


def run_step(step_name: str, cmd: list[str], cwd: Path) -> None:
    print(f"\n=== {step_name} ===")
    print("Running:", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=cwd)
    if completed.returncode != 0:
        raise SystemExit(f"Step failed: {step_name} (exit code {completed.returncode})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full 311 reporting pipeline: dataset profile, summary analysis, and charts/dashboard."
        )
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for each step (defaults to current interpreter).",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Optional row limit passed to analysis and plotting scripts.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Top N categories passed to analysis and plotting scripts.",
    )
    parser.add_argument(
        "--analysis-json",
        default="aea/output/generated_json/summary_full.json",
        help="Output JSON path for analyze_311_summary.py.",
    )
    parser.add_argument(
        "--charts-output-dir",
        default="aea/output/charts",
        help="Output directory for generated chart images.",
    )
    parser.add_argument(
        "--dashboard-html",
        default="docs/summary_dashboard.html",
        help="Output path for dashboard HTML file.",
    )
    parser.add_argument(
        "--map-output",
        default="docs/community_map.html",
        help="Output path for the community map HTML file.",
    )
    parser.add_argument(
        "--map-mode",
        choices=["heatmap", "cluster", "both"],
        default="both",
        help="Map mode passed to map_311_communities_v2.py (default: both).",
    )
    parser.add_argument(
        "--skip-profile",
        action="store_true",
        help="Skip datasetProfile.py and only run summary analysis + charts/dashboard.",
    )
    parser.add_argument(
        "--skip-map",
        action="store_true",
        help="Skip map_311_communities_v2.py.",
    )
    parser.add_argument(
        "--skip-weather",
        action="store_true",
        help="Skip weather_311_analysis.py.",
    )
    parser.add_argument(
        "--weather-output",
        default="docs/weather_311_analysis.html",
        help="Output path for the weather analysis HTML file.",
    )
    parser.add_argument(
        "--top-services",
        type=int,
        default=20,
        help="Top N services shown in weather analysis charts (default: 20).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    python_cmd = args.python

    profile_cmd = [python_cmd, "aea/scripts/datasetProfile.py"]

    analysis_cmd = [
        python_cmd,
        "aea/scripts/analyze_311_summary.py",
        "--top-n",
        str(args.top_n),
        "--output-json",
        args.analysis_json,
    ]
    if args.nrows is not None:
        analysis_cmd.extend(["--nrows", str(args.nrows)])

    plots_cmd = [
        python_cmd,
        "aea/scripts/plot_311_summary.py",
        "--top-n",
        str(args.top_n),
        "--output-dir",
        args.charts_output_dir,
        "--html-output",
        args.dashboard_html,
    ]
    if args.nrows is not None:
        plots_cmd.extend(["--nrows", str(args.nrows)])

    map_cmd = [
        python_cmd,
        "aea/scripts/map_311_communities_v2.py",
        "--output",
        args.map_output,
        "--mode",
        args.map_mode,
    ]
    if args.nrows is not None:
        map_cmd.extend(["--nrows", str(args.nrows)])

    weather_cmd = [
        python_cmd,
        "aea/scripts/weather_311_analysis.py",
        "--output",
        args.weather_output,
        "--top-services",
        str(args.top_services),
    ]
    if args.nrows is not None:
        weather_cmd.extend(["--nrows", str(args.nrows)])

    print("Pipeline starting from:", project_root)
    step_idx = 1
    if args.skip_profile:
        print("\n=== 1) Skip ydata profile ===")
        print("Skipping: aea/scripts/datasetProfile.py")
    else:
        run_step(f"{step_idx}) Generate ydata profile", profile_cmd, project_root)
        step_idx += 1

    run_step(f"{step_idx}) Generate summary analysis", analysis_cmd, project_root)
    step_idx += 1
    run_step(f"{step_idx}) Generate charts + dashboard", plots_cmd, project_root)
    step_idx += 1

    if args.skip_map:
        print(f"\n=== {step_idx}) Skip community map ===")
        print("Skipping: aea/scripts/map_311_communities_v2.py")
    else:
        run_step(f"{step_idx}) Generate community map", map_cmd, project_root)
    step_idx += 1

    if args.skip_weather:
        print(f"\n=== {step_idx}) Skip weather analysis ===")
        print("Skipping: aea/scripts/weather_311_analysis.py")
    else:
        run_step(f"{step_idx}) Generate weather analysis", weather_cmd, project_root)

    print("\nPipeline completed successfully.")
    print("Artifacts:")
    if args.skip_profile:
        print("- Profile HTML: skipped")
    else:
        print("- Profile HTML: docs/report.html")
    print(f"- Analysis JSON: {args.analysis_json}")
    print(f"- Charts folder: {args.charts_output_dir}")
    print(f"- Dashboard HTML: {args.dashboard_html}")
    if args.skip_map:
        print("- Community map: skipped")
    else:
        print(f"- Community map: {args.map_output}")
    if args.skip_weather:
        print("- Weather analysis: skipped")
    else:
        print(f"- Weather analysis: {args.weather_output}")


if __name__ == "__main__":
    main()
