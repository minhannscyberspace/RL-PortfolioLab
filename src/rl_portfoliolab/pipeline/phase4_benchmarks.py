from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from rl_portfoliolab.benchmarks.policies import (
    equal_weight,
    inverse_vol_weights,
    mean_variance_weights,
    momentum_weights,
)
from rl_portfoliolab.evaluation.metrics import cvar, max_drawdown, sharpe, turnover_from_weights
from rl_portfoliolab.pipeline.phase2_env import make_env_from_configs, load_phase1_features_json, load_phase2_env_config
from rl_portfoliolab.utils.seeding import set_seed


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    lookback: int = 20
    ridge: float = 1e-6


@dataclass(frozen=True)
class Phase4Config:
    seed: int
    phase2_env_config_path: str
    benchmarks: list[BenchmarkSpec]
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


def load_phase4_config(path: str | Path) -> Phase4Config:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    root = _require_mapping(raw, "root")

    bm_raw = root.get("benchmarks")
    if not isinstance(bm_raw, list) or not bm_raw:
        raise ValueError("benchmarks must be a non-empty list")

    benchmarks: list[BenchmarkSpec] = []
    for i, item in enumerate(bm_raw):
        m = _require_mapping(item, f"benchmarks[{i}]")
        name = _require_str(m.get("name"), f"benchmarks[{i}].name")
        lookback = int(m.get("lookback", 20))
        ridge = float(m.get("ridge", 1e-6))
        benchmarks.append(BenchmarkSpec(name=name, lookback=lookback, ridge=ridge))

    out = _require_mapping(root.get("output"), "output")
    return Phase4Config(
        seed=_require_int(root.get("seed"), "seed"),
        phase2_env_config_path=_require_str(root.get("phase2_env_config_path"), "phase2_env_config_path"),
        benchmarks=benchmarks,
        out_dir=_require_str(out.get("out_dir"), "output.out_dir"),
        run_name=_require_str(out.get("run_name"), "output.run_name"),
    )


def _run_policy_episode(
    *,
    policy: str,
    lookback: int,
    ridge: float,
    phase2_env_config_path: str,
) -> dict[str, Any]:
    # Build base env
    env = make_env_from_configs(phase2_yaml_path=phase2_env_config_path)
    obs = env.reset()
    n = len(obs["assets"])

    # Load underlying features (returns/vol) for momentum/vol policies
    phase2_cfg = load_phase2_env_config(phase2_env_config_path)
    phase1 = load_phase1_features_json(phase2_cfg.phase1_features_path)
    returns_series = phase1["returns"]  # (T, N)
    vol_series = phase1["volatility"]  # (T, N)
    cov_series = phase1["covariance"]  # (T, N, N)

    total_reward = 0.0
    series = []
    weights_series: list[list[Optional[float]]] = []
    per_step_portfolio_returns: list[Optional[float]] = []
    equity_series: list[Optional[float]] = []

    done = False
    while not done:
        t = int(obs["t"])

        if policy == "equal_weight":
            action = equal_weight(n)
        elif policy == "momentum":
            start = max(0, t - lookback)
            window = returns_series[start:t] if t > 0 else returns_series[0:1]
            action = momentum_weights(window)
        elif policy == "inverse_vol":
            vol_row = vol_series[t] if t < len(vol_series) else vol_series[-1]
            action = inverse_vol_weights(vol_row)
        elif policy == "mean_variance":
            start = max(0, t - lookback)
            window = returns_series[start:t] if t > 0 else returns_series[0:1]
            cov_row = cov_series[t] if t < len(cov_series) else cov_series[-1]
            action = mean_variance_weights(returns_window=window, cov_row=cov_row, ridge=ridge)
        else:
            raise ValueError(f"Unknown policy: {policy}")

        obs, reward, done, info = env.step(action)
        total_reward += float(reward)

        weights_series.append(list(obs["portfolio"]["weights"]))
        per_step_portfolio_returns.append(float(info.get("portfolio_return", 0.0)))
        equity_series.append(float(info.get("equity_after", 0.0)))

        series.append(
            {
                "t": int(info.get("t", t)),
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

    summary = {
        "steps": len(series),
        "total_reward": total_reward,
        "final_equity": equity_series[-1] if equity_series else None,
        "sharpe": sharpe(per_step_portfolio_returns),
        "max_drawdown": max_drawdown(equity_series),
        "cvar_5": cvar(per_step_portfolio_returns, alpha=0.05),
        "avg_turnover": turnover_from_weights(weights_series),
    }

    return {"policy": policy, "lookback": lookback, "ridge": ridge, "summary": summary, "series": series}


def run_phase4(cfg: Phase4Config) -> Path:
    set_seed(cfg.seed)

    out_dir = Path(cfg.out_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for bm in cfg.benchmarks:
        res = _run_policy_episode(
            policy=bm.name,
            lookback=bm.lookback,
            ridge=bm.ridge,
            phase2_env_config_path=cfg.phase2_env_config_path,
        )
        results.append(res)
        (out_dir / f"{bm.name}.eval.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    # Comparison table
    table = []
    for res in results:
        s = res["summary"]
        table.append(
            {
                "policy": res["policy"],
                "final_equity": s["final_equity"],
                "total_reward": s["total_reward"],
                "sharpe": s["sharpe"],
                "max_drawdown": s["max_drawdown"],
                "cvar_5": s["cvar_5"],
                "avg_turnover": s["avg_turnover"],
            }
        )

    comparison_path = out_dir / "comparison.json"
    comparison_path.write_text(json.dumps(table, indent=2), encoding="utf-8")
    return comparison_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4 benchmark comparisons.")
    parser.add_argument("--config", required=True, help="Path to configs/phase4_benchmarks.yaml")
    args = parser.parse_args()

    cfg = load_phase4_config(args.config)
    out = run_phase4(cfg)
    print(f"Wrote comparison artifact: {out}")


if __name__ == "__main__":
    main()

