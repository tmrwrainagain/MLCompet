"""
Module B — main entry point.

Usage:
    python main.py                  # run clustering + quality assessment + report
    python main.py --cluster        # clustering only
    python main.py --quality        # quality metrics only
    python main.py --report         # report only
    python main.py --dashboard      # launch Streamlit dashboard
    python main.py --all            # full pipeline then launch dashboard
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def cmd_cluster():
    print("\n[Module B] Running clustering...")
    from clustering import run_clustering
    df = run_clustering()
    print(f"  Clustered {len(df)} materials.")
    return df


def cmd_quality(df=None):
    print("\n[Module B] Evaluating clustering quality...")
    if df is None:
        from clustering import load_materials_df
        df = load_materials_df()
    if df.empty:
        print("  No materials — skipping quality assessment.")
        return {}, df
    from quality_metrics import evaluate_clustering
    results = evaluate_clustering(df)
    # Print summary
    for key in ("parallel_cluster", "sequential_cluster", "complexity_cluster"):
        if key in results:
            m = results[key].get("metrics", {})
            print(f"  {results[key]['name']}: "
                  f"Silhouette={m.get('silhouette', float('nan')):.3f}  "
                  f"DB={m.get('davies_bouldin', float('nan')):.3f}  "
                  f"CH={m.get('calinski_harabasz', float('nan')):.1f}")
    return results, df


def cmd_report(df=None, quality_results=None):
    print("\n[Module B] Generating report...")
    if df is None:
        from clustering import load_materials_df
        df = load_materials_df()
    if quality_results is None:
        quality_results = {}
    from report_generator import generate_report
    path = generate_report(df, quality_results)
    print(f"  Report: {path}")


def cmd_dashboard():
    print("\n[Module B] Launching Streamlit dashboard...")
    print("  URL: http://localhost:8501")
    print("  Users: admin/admin123  teacher/teacher123  student/student123")
    print("  Press Ctrl+C to stop.\n")
    dashboard_path = Path(__file__).parent / "dashboard.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run",
                    str(dashboard_path), "--server.headless", "true"])


def main():
    parser = argparse.ArgumentParser(description="Module B — Educational Materials Analytics")
    parser.add_argument("--cluster",   action="store_true", help="Run clustering")
    parser.add_argument("--quality",   action="store_true", help="Evaluate clustering quality")
    parser.add_argument("--report",    action="store_true", help="Generate HTML report")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard")
    parser.add_argument("--all",       action="store_true", help="Full pipeline + dashboard")
    args = parser.parse_args()

    sep = "=" * 60
    print(sep)
    print("  MODULE B — Analytics & Visualisation")
    print(sep)

    if args.all:
        df               = cmd_cluster()
        quality_results, df = cmd_quality(df)
        cmd_report(df, quality_results)
        cmd_dashboard()
        return

    if args.cluster:
        cmd_cluster()
        return

    if args.quality:
        cmd_quality()
        return

    if args.report:
        cmd_report()
        return

    if args.dashboard:
        cmd_dashboard()
        return

    # Default: full pipeline without launching dashboard
    df                  = cmd_cluster()
    quality_results, df = cmd_quality(df)
    cmd_report(df, quality_results)

    print(f"\n{sep}")
    print("  Module B pipeline complete!")
    print(f"  To launch dashboard: python main.py --dashboard")
    print(sep)


if __name__ == "__main__":
    main()
