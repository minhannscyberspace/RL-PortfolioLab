from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from rl_portfoliolab.pipeline.phase2_env import make_env_from_configs
from rl_portfoliolab.utils.seeding import set_seed


@dataclass(frozen=True)
class Phase3EvalConfig:
    seed: int
    phase2_env_config_path: str
    algo: str
    model_path: str
    max_steps: int
    deterministic: bool
    out_dir: str
    run_name: str


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


def _require_bool(x: Any, name: str) -> bool:
    if not isinstance(x, bool):
        raise ValueError(f"`{name}` must be a bool.")
    return x


def load_phase3_eval_config(path: str | Path) -> Phase3EvalConfig:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    root = _require_mapping(raw, "root")

    model = _require_mapping(root.get("model"), "model")
    eval_cfg = _require_mapping(root.get("eval"), "eval")
    output = _require_mapping(root.get("output"), "output")

    algo = _require_str(model.get("algo"), "model.algo")
    if algo not in ("PPO", "SAC"):
        raise ValueError("model.algo must be 'PPO' or 'SAC'")

    return Phase3EvalConfig(
        seed=_require_int(root.get("seed"), "seed"),
        phase2_env_config_path=_require_str(root.get("phase2_env_config_path"), "phase2_env_config_path"),
        algo=algo,
        model_path=_require_str(model.get("model_path"), "model.model_path"),
        max_steps=_require_int(eval_cfg.get("max_steps"), "eval.max_steps"),
        deterministic=_require_bool(eval_cfg.get("deterministic"), "eval.deterministic"),
        out_dir=_require_str(output.get("out_dir"), "output.out_dir"),
        run_name=_require_str(output.get("run_name"), "output.run_name"),
    )


def _require_sb3_model_loader(algo: str):
    try:
        if algo == "PPO":
            from stable_baselines3 import PPO as Model  # type: ignore
        else:
            from stable_baselines3 import SAC as Model  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Stable-Baselines3 failed to import. Install `stable-baselines3` (and its deps) and retry."
        ) from e
    return Model


def evaluate_once(cfg: Phase3EvalConfig) -> Path:
    """
    Load the trained SB3 model and run a single rollout.

    Writes a JSON artifact containing:
    - summary metrics (total reward, final equity, avg turnover/costs)
    - per-step series from env `info`
    """
    set_seed(cfg.seed)

    # Create env and Gymnasium wrapper to match training.
    try:
        import gymnasium as gym  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("gymnasium failed to import. Install `gymnasium` and retry.") from e

    from rl_portfoliolab.envs.sb3_adapter import make_gymnasium_env

    base_env = make_env_from_configs(phase2_yaml_path=cfg.phase2_env_config_path)
    env = make_gymnasium_env(base_env=base_env, gym=gym)

    Model = _require_sb3_model_loader(cfg.algo)
    model = Model.load(cfg.model_path)

    obs, _ = env.reset()

    total_reward = 0.0
    steps = 0
    series = []

    while steps < cfg.max_steps:
        action, _state = model.predict(obs, deterministic=cfg.deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1

        # Persist selected info keys for analysis.
        series.append(
            {
                "t": int(info.get("t", steps - 1)),
                "turnover": float(info.get("turnover", 0.0)),
                "cost": float(info.get("cost", 0.0)),
                "slippage": float(info.get("slippage", 0.0)),
                "turnover_penalty": float(info.get("turnover_penalty", 0.0)),
                "portfolio_return": float(info.get("portfolio_return", 0.0)),
                "equity_before": float(info.get("equity_before", 0.0)),
                "equity_after": float(info.get("equity_after", 0.0)),
                "reward": float(reward),
            }
        )

        if terminated or truncated:
            break

    final_equity = series[-1]["equity_after"] if series else None
    avg_turnover = (sum(x["turnover"] for x in series) / len(series)) if series else 0.0
    avg_cost = (sum(x["cost"] for x in series) / len(series)) if series else 0.0

    out_dir = Path(cfg.out_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval.json"

    payload = {
        "config": {
            "seed": cfg.seed,
            "phase2_env_config_path": cfg.phase2_env_config_path,
            "algo": cfg.algo,
            "model_path": cfg.model_path,
            "max_steps": cfg.max_steps,
            "deterministic": cfg.deterministic,
        },
        "summary": {
            "steps": steps,
            "total_reward": total_reward,
            "final_equity": final_equity,
            "avg_turnover": avg_turnover,
            "avg_cost": avg_cost,
        },
        "series": series,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 evaluation rollout.")
    parser.add_argument("--config", required=True, help="Path to configs/phase3_eval.yaml")
    args = parser.parse_args()

    cfg = load_phase3_eval_config(args.config)
    out = evaluate_once(cfg)
    print(f"Wrote eval artifact: {out}")


if __name__ == "__main__":
    main()

