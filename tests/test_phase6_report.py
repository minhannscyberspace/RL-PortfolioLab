from __future__ import annotations

import json
from pathlib import Path


def test_phase6_report_writes_files(tmp_path: Path) -> None:
    # Minimal fake inputs
    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            [
                {
                    "policy": "equal_weight",
                    "final_equity": 100.0,
                    "total_reward": 0.0,
                    "sharpe": 0.0,
                    "max_drawdown": 0.0,
                    "cvar_5": 0.0,
                    "avg_turnover": 0.0,
                }
            ]
        ),
        encoding="utf-8",
    )

    stress_path = tmp_path / "stress_summary.json"
    stress_path.write_text(
        json.dumps(
            [
                {
                    "policy": "equal_weight",
                    "regimes": {
                        "calm": {"steps": 10, "sharpe": 0.1, "max_drawdown": 0.01, "cvar_5": -0.02, "final_equity": 101.0}
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    wf_path = tmp_path / "walkforward_summary.json"
    wf_path.write_text(
        json.dumps(
            {
                "n": 100,
                "splits": [{"train_start": 0, "train_end": 60, "test_start": 60, "test_end": 90}],
                "folds": [
                    {
                        "fold": 0,
                        "split": {"train_start": 0, "train_end": 60, "test_start": 60, "test_end": 90},
                        "results": [
                            {"policy": "equal_weight", "summary": {"sharpe": 0.1}},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    from rl_portfoliolab.pipeline.phase6_report import Phase6Config, run_phase6

    out_dir = tmp_path / "reports"
    cfg = Phase6Config(
        seed=1,
        benchmarks_comparison_path=str(comparison_path),
        stress_summary_path=str(stress_path),
        walkforward_summary_path=str(wf_path),
        rl_eval_path=None,
        data_config_path=None,
        phase2_env_config_path=None,
        out_dir=str(out_dir),
        run_name="x",
    )

    md_path, html_path = run_phase6(cfg)
    assert md_path.exists()
    assert html_path.exists()
