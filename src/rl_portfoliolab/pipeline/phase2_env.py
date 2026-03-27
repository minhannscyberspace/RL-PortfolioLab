from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from rl_portfoliolab.envs.portfolio_env import (
    PortfolioAllocationEnv,
    PortfolioEnvConfig,
)
from rl_portfoliolab.utils.seeding import set_seed


@dataclass(frozen=True)
class Phase2EnvYaml:
    seed: int
    phase1_features_path: str
    env: PortfolioEnvConfig


def _require_mapping(x: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(x, Mapping):
        raise ValueError(f"`{name}` must be an object/mapping.")
    return x


def _require_str(x: Any, name: str) -> str:
    if not isinstance(x, str):
        raise ValueError(f"`{name}` must be a string.")
    return x


def _require_float(x: Any, name: str) -> float:
    if not isinstance(x, (int, float)):
        raise ValueError(f"`{name}` must be a number.")
    return float(x)


def load_phase2_env_config(path: str | Path) -> Phase2EnvYaml:
    p = Path(path)
    raw = yaml.safe_load(p.read_text())
    root = _require_mapping(raw, "root")

    seed = int(root.get("seed"))
    phase1_features_path = _require_str(root.get("phase1_features_path"), "phase1_features_path")

    env_raw = _require_mapping(root.get("env"), "env")
    env_cfg = PortfolioEnvConfig(
        initial_cash=_require_float(env_raw.get("initial_cash"), "env.initial_cash"),
        min_weight=_require_float(env_raw.get("min_weight"), "env.min_weight"),
        max_weight=_require_float(env_raw.get("max_weight"), "env.max_weight"),
        max_gross_exposure=_require_float(env_raw.get("max_gross_exposure"), "env.max_gross_exposure"),
        transaction_cost_rate=_require_float(env_raw.get("transaction_cost_rate"), "env.transaction_cost_rate"),
        slippage_rate=_require_float(env_raw.get("slippage_rate"), "env.slippage_rate"),
        turnover_penalty=_require_float(env_raw.get("turnover_penalty"), "env.turnover_penalty"),
    )

    return Phase2EnvYaml(seed=seed, phase1_features_path=phase1_features_path, env=env_cfg)


def load_phase1_features_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    root = _require_mapping(raw, "phase1_features_json")

    # These are nested lists and may contain nulls for missing values.
    assets = root.get("assets")
    returns = root.get("returns")
    volatility = root.get("volatility")
    covariance = root.get("covariance")

    if not isinstance(assets, list) or not all(isinstance(a, str) for a in assets):
        raise ValueError("phase1 features `assets` must be a list of strings.")
    if not isinstance(returns, list) or not isinstance(volatility, list) or not isinstance(covariance, list):
        raise ValueError("phase1 features must include returns/volatility/covariance lists.")

    return {
        "assets": assets,
        "returns": returns,
        "volatility": volatility,
        "covariance": covariance,
    }


def make_env_from_configs(*, phase2_yaml_path: str | Path) -> PortfolioAllocationEnv:
    cfg = load_phase2_env_config(phase2_yaml_path)
    set_seed(cfg.seed)

    phase1 = load_phase1_features_json(cfg.phase1_features_path)
    return PortfolioAllocationEnv(
        returns=phase1["returns"],
        volatility=phase1["volatility"],
        covariance=phase1["covariance"],
        assets=phase1["assets"],
        config=cfg.env,
    )

