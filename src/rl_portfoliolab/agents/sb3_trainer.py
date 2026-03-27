from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from rl_portfoliolab.pipeline.phase2_env import make_env_from_configs
from rl_portfoliolab.utils.seeding import set_seed


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    phase2_env_config_path: str
    algo_name: Literal["PPO", "SAC"]
    policy: str
    total_timesteps: int
    model_dir: str
    run_name: str


class MissingOptionalDependency(RuntimeError):
    pass


def _require_sb3() -> tuple[Any, Any]:
    """
    Import Stable-Baselines3 lazily, so the repo stays importable even when SB3 isn't installed.
    """
    try:
        from stable_baselines3 import PPO, SAC  # type: ignore
    except Exception as e:  # pragma: no cover
        raise MissingOptionalDependency(
            "Stable-Baselines3 is not installed or failed to import. "
            "Install it in a working Python environment and retry."
        ) from e
    return PPO, SAC


def _require_gymnasium() -> Any:
    try:
        import gymnasium as gym  # type: ignore
    except Exception as e:  # pragma: no cover
        raise MissingOptionalDependency(
            "gymnasium is not installed or failed to import. Install gymnasium and retry."
        ) from e
    return gym


def train_with_sb3(cfg: TrainConfig) -> Path:
    """
    Train a policy using Stable-Baselines3 PPO/SAC.

    Note: this requires a functional NumPy/Torch stack. In this sandbox environment,
    NumPy has been crashing on import, so this is expected to be run locally in a clean env.
    """
    set_seed(cfg.seed)

    # Fail fast with a clear message if NumPy can't import (segfaults are possible in some envs).
    # We intentionally do NOT import NumPy here. In some environments (including this one),
    # importing NumPy can segfault the process, which is not catchable in Python.
    # Instead, we leave dependency verification to the user's chosen runtime.

    gym = _require_gymnasium()
    PPO, SAC = _require_sb3()

    # Wrap our dependency-light env with a Gymnasium adapter.
    from rl_portfoliolab.envs.sb3_adapter import make_gymnasium_env

    base_env = make_env_from_configs(phase2_yaml_path=cfg.phase2_env_config_path)
    env = make_gymnasium_env(base_env=base_env, gym=gym)

    model_cls = PPO if cfg.algo_name == "PPO" else SAC
    model = model_cls(cfg.policy, env, verbose=1, seed=cfg.seed)
    model.learn(total_timesteps=cfg.total_timesteps)

    out_dir = Path(cfg.model_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{cfg.algo_name.lower()}_model.zip"
    model.save(str(model_path))
    return model_path

