from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from rl_portfoliolab.agents.sb3_trainer import TrainConfig, train_with_sb3


def _require_mapping(x: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(x, Mapping):
        raise ValueError(f"`{name}` must be an object/mapping.")
    return x


def _require_str(x: Any, name: str) -> str:
    if not isinstance(x, str):
        raise ValueError(f"`{name}` must be a string.")
    return x


def _require_int(x: Any, name: str) -> int:
    if not isinstance(x, int):
        raise ValueError(f"`{name}` must be an int.")
    return x


def load_phase3_train_config(path: str | Path) -> TrainConfig:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    root = _require_mapping(raw, "root")

    algo = _require_mapping(root.get("algo"), "algo")
    output = _require_mapping(root.get("output"), "output")

    algo_name = _require_str(algo.get("name"), "algo.name")
    if algo_name not in ("PPO", "SAC"):
        raise ValueError("algo.name must be 'PPO' or 'SAC'")

    return TrainConfig(
        seed=_require_int(root.get("seed"), "seed"),
        phase2_env_config_path=_require_str(root.get("phase2_env_config_path"), "phase2_env_config_path"),
        algo_name=algo_name,  # type: ignore[arg-type]
        policy=_require_str(algo.get("policy"), "algo.policy"),
        total_timesteps=_require_int(algo.get("total_timesteps"), "algo.total_timesteps"),
        model_dir=_require_str(output.get("model_dir"), "output.model_dir"),
        run_name=_require_str(output.get("run_name"), "output.run_name"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 training (SB3 PPO/SAC).")
    parser.add_argument("--config", required=True, help="Path to configs/phase3_train.yaml")
    args = parser.parse_args()

    cfg = load_phase3_train_config(args.config)
    model_path = train_with_sb3(cfg)
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()

