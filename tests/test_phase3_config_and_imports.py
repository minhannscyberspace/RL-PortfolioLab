from __future__ import annotations

from pathlib import Path


def test_phase3_config_parses(tmp_path: Path) -> None:
    cfg_path = tmp_path / "phase3.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "seed: 1",
                "phase2_env_config_path: configs/phase2_env.yaml",
                "algo:",
                "  name: PPO",
                "  total_timesteps: 10",
                "  policy: MlpPolicy",
                "output:",
                "  model_dir: artifacts/phase3/models",
                "  run_name: test",
                "",
            ]
        ),
        encoding="utf-8",
    )

    from rl_portfoliolab.pipeline.phase3_train import load_phase3_train_config

    cfg = load_phase3_train_config(cfg_path)
    assert cfg.algo_name == "PPO"
    assert cfg.total_timesteps == 10
    assert cfg.run_name == "test"


def test_phase3_modules_import_without_sb3_installed() -> None:
    # These imports must not require stable-baselines3/gymnasium at import time.
    import rl_portfoliolab.agents.sb3_trainer as _  # noqa: F401
    import rl_portfoliolab.envs.sb3_adapter as _2  # noqa: F401

