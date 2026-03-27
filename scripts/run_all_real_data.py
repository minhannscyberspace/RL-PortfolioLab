from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run_step(repo_root: Path, label: str, args: list[str]) -> None:
    print(f"\n=== {label} ===")
    print(" ".join(args))
    subprocess.run(args, cwd=str(repo_root), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full real-data pipeline: fetch -> phase1..7 -> report.\n"
            "This script orchestrates existing phase scripts in one command."
        )
    )
    parser.add_argument("--data-config", default="configs/data_stooq.yaml", help="Path to data fetch config")
    parser.add_argument("--phase1-config", default="configs/phase1_wide.yaml", help="Path to phase1 config")
    parser.add_argument("--phase2-config", default="configs/phase2_env.yaml", help="Path to phase2 config")
    parser.add_argument("--phase3-train-config", default="configs/phase3_train.yaml", help="Path to phase3 train config")
    parser.add_argument("--phase3-eval-config", default="configs/phase3_eval.yaml", help="Path to phase3 eval config")
    parser.add_argument(
        "--phase4-config", default="configs/phase4_benchmarks.yaml", help="Path to phase4 benchmarks config"
    )
    parser.add_argument("--phase5-config", default="configs/phase5_stress.yaml", help="Path to phase5 stress config")
    parser.add_argument("--phase6-config", default="configs/phase6_report.yaml", help="Path to phase6 report config")
    parser.add_argument(
        "--phase7-config", default="configs/phase7_walkforward.yaml", help="Path to phase7 walk-forward config"
    )
    parser.add_argument(
        "--phase1-csv",
        default="data/raw/prices_wide.csv",
        help="Wide CSV path to pass into phase1 after fetch",
    )
    parser.add_argument(
        "--phase1-out",
        default="artifacts/phase1",
        help="Output directory for phase1 feature artifacts",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    python = sys.executable

    plan = [
        (
            "Fetch real market data (wide CSV)",
            [python, "scripts/fetch_prices_wide.py", "--config", args.data_config],
        ),
        (
            "Phase 1 - data alignment and features",
            [
                python,
                "scripts/run_phase1.py",
                "--config",
                args.phase1_config,
                "--csv",
                args.phase1_csv,
                "--out",
                args.phase1_out,
            ],
        ),
        (
            "Phase 2 - environment smoke run",
            [python, "scripts/run_phase2_env.py", "--config", args.phase2_config],
        ),
        (
            "Phase 3 - RL training (SB3)",
            [python, "scripts/run_phase3_train.py", "--config", args.phase3_train_config],
        ),
        (
            "Phase 3 - RL evaluation",
            [python, "scripts/run_phase3_eval.py", "--config", args.phase3_eval_config],
        ),
        (
            "Phase 4 - benchmark suite",
            [python, "scripts/run_phase4_benchmarks.py", "--config", args.phase4_config],
        ),
        (
            "Phase 5 - stress/regime slicing",
            [python, "scripts/run_phase5_stress.py", "--config", args.phase5_config],
        ),
        (
            "Phase 7 - walk-forward OOS benchmarks",
            [python, "scripts/run_phase7_walkforward.py", "--config", args.phase7_config],
        ),
        (
            "Phase 6 - auto-report generation",
            [python, "scripts/run_phase6_report.py", "--config", args.phase6_config],
        ),
    ]

    for label, cmd in plan:
        _run_step(repo_root=repo_root, label=label, args=cmd)

    print("\nPipeline completed successfully.")
    print("Report: reports/baseline_report/report.html")


if __name__ == "__main__":
    main()
