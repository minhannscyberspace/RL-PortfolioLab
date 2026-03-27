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
from rl_portfoliolab.envs.portfolio_env import PortfolioAllocationEnv
from rl_portfoliolab.evaluation.metrics import cvar, max_drawdown, sharpe, turnover_from_weights
from rl_portfoliolab.evaluation.splits import WalkForwardSplit, make_walkforward_splits
from rl_portfoliolab.pipeline.phase2_env import load_phase1_features_json, load_phase2_env_config
from rl_portfoliolab.utils.seeding import set_seed


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    lookback: int = 20
    ridge: float = 1e-6


@dataclass(frozen=True)
class Phase7Config:
    seed: int
    phase2_env_config_path: str
    train_size: int
    test_size: int
    step_size: int
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


def load_phase7_config(path: str | Path) -> Phase7Config:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    root = _require_mapping(raw, "root")

    splits = _require_mapping(root.get("splits"), "splits")
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
    return Phase7Config(
        seed=_require_int(root.get("seed"), "seed"),
        phase2_env_config_path=_require_str(root.get("phase2_env_config_path"), "phase2_env_config_path"),
        train_size=_require_int(splits.get("train_size"), "splits.train_size"),
        test_size=_require_int(splits.get("test_size"), "splits.test_size"),
        step_size=_require_int(splits.get("step_size"), "splits.step_size"),
        benchmarks=benchmarks,
        out_dir=_require_str(out.get("out_dir"), "output.out_dir"),
        run_name=_require_str(out.get("run_name"), "output.run_name"),
    )


def _make_env_slice(
    *,
    phase2_env_config_path: str,
    test_start: int,
    test_end: int,
) -> tuple[PortfolioAllocationEnv, dict[str, Any]]:
    phase2_cfg = load_phase2_env_config(phase2_env_config_path)
    phase1 = load_phase1_features_json(phase2_cfg.phase1_features_path)

    assets = phase1["assets"]
    returns_full = phase1["returns"]
    vol_full = phase1["volatility"]
    cov_full = phase1["covariance"]

    returns = returns_full[test_start:test_end]
    vol = vol_full[test_start:test_end]
    cov = cov_full[test_start:test_end]

    env = PortfolioAllocationEnv(
        returns=returns,
        volatility=vol,
        covariance=cov,
        assets=assets,
        config=phase2_cfg.env,
    )
    return env, phase1


def _policy_action(
    *,
    policy: str,
    lookback: int,
    ridge: float,
    global_t: int,
    n_assets: int,
    returns_full: list[list[Optional[float]]],
    vol_full: list[list[Optional[float]]],
    cov_full: list[list[list[Optional[float]]]],
) -> list[float]:
    if policy == "equal_weight":
        return equal_weight(n_assets)
    if policy == "momentum":
        start = max(0, global_t - lookback)
        window = returns_full[start:global_t] if global_t > 0 else returns_full[0:1]
        return momentum_weights(window)
    if policy == "inverse_vol":
        vol_row = vol_full[global_t] if global_t < len(vol_full) else vol_full[-1]
        return inverse_vol_weights(vol_row)
    if policy == "mean_variance":
        start = max(0, global_t - lookback)
        window = returns_full[start:global_t] if global_t > 0 else returns_full[0:1]
        cov_row = cov_full[global_t] if global_t < len(cov_full) else cov_full[-1]
        return mean_variance_weights(returns_window=window, cov_row=cov_row, ridge=ridge)
    raise ValueError(f"Unknown policy: {policy}")


def _run_oos_for_policy(
    *,
    bm: BenchmarkSpec,
    split: WalkForwardSplit,
    phase2_env_config_path: str,
) -> dict[str, Any]:
    env, phase1 = _make_env_slice(
        phase2_env_config_path=phase2_env_config_path,
        test_start=split.test_start,
        test_end=split.test_end,
    )
    obs = env.reset()
    n = len(obs["assets"])

    returns_full = phase1["returns"]
    vol_full = phase1["volatility"]
    cov_full = phase1["covariance"]

    total_reward = 0.0
    series = []
    weights_series: list[list[Optional[float]]] = []
    per_step_portfolio_returns: list[Optional[float]] = []
    equity_series: list[Optional[float]] = []

    done = False
    while not done:
        local_t = int(obs["t"])
        global_t = split.test_start + local_t

        action = _policy_action(
            policy=bm.name,
            lookback=bm.lookback,
            ridge=bm.ridge,
            global_t=global_t,
            n_assets=n,
            returns_full=returns_full,
            vol_full=vol_full,
            cov_full=cov_full,
        )

        obs, reward, done, info = env.step(action)
        total_reward += float(reward)

        weights_series.append(list(obs["portfolio"]["weights"]))
        per_step_portfolio_returns.append(float(info.get("portfolio_return", 0.0)))
        equity_series.append(float(info.get("equity_after", 0.0)))

        series.append(
            {
                "local_t": local_t,
                "global_t": global_t,
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

    return {
        "policy": bm.name,
        "lookback": bm.lookback,
        "ridge": bm.ridge,
        "split": {
            "train_start": split.train_start,
            "train_end": split.train_end,
            "test_start": split.test_start,
            "test_end": split.test_end,
        },
        "summary": summary,
        "series": series,
    }


def run_phase7(cfg: Phase7Config) -> Path:
    set_seed(cfg.seed)

    # Determine total length from phase1 features.
    phase2_cfg = load_phase2_env_config(cfg.phase2_env_config_path)
    phase1 = load_phase1_features_json(phase2_cfg.phase1_features_path)
    n = len(phase1["returns"])

    splits = make_walkforward_splits(
        n=n,
        train_size=cfg.train_size,
        test_size=cfg.test_size,
        step_size=cfg.step_size,
    )

    out_dir = Path(cfg.out_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for split_idx, split in enumerate(splits):
        fold_results = []
        for bm in cfg.benchmarks:
            res = _run_oos_for_policy(bm=bm, split=split, phase2_env_config_path=cfg.phase2_env_config_path)
            fold_results.append(res)
            (out_dir / f"fold{split_idx}_{bm.name}.oos.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

        all_results.append({"fold": split_idx, "split": split, "results": fold_results})

    summary_path = out_dir / "walkforward_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "n": n,
                "splits": [s.__dict__ for s in splits],
                "folds": all_results,
            },
            indent=2,
            default=lambda o: o.__dict__,
        ),
        encoding="utf-8",
    )
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 7 walk-forward OOS evaluation (benchmarks).")
    parser.add_argument("--config", required=True, help="Path to configs/phase7_walkforward.yaml")
    args = parser.parse_args()

    cfg = load_phase7_config(args.config)
    out = run_phase7(cfg)
    print(f"Wrote walk-forward summary: {out}")


if __name__ == "__main__":
    main()

