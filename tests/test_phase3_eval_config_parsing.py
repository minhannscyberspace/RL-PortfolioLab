from __future__ import annotations

from pathlib import Path


def test_phase3_eval_config_parses(tmp_path: Path) -> None:
    cfg_path = tmp_path / "phase3_eval.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "seed: 1",
                "phase2_env_config_path: configs/phase2_env.yaml",
                "model:",
                "  algo: PPO",
                "  model_path: artifacts/phase3/models/baseline/ppo_model.zip",
                "eval:",
                "  max_steps: 10",
                "  deterministic: true",
                "output:",
                "  out_dir: artifacts/phase3/eval",
                "  run_name: test_eval",
                "",
            ]
        ),
        encoding="utf-8",
    )

    from rl_portfoliolab.pipeline.phase3_eval import load_phase3_eval_config

    cfg = load_phase3_eval_config(cfg_path)
    assert cfg.algo == "PPO"
    assert cfg.max_steps == 10
    assert cfg.deterministic is True

