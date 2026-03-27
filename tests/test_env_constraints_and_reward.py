from __future__ import annotations

import math

from rl_portfoliolab.envs.portfolio_env import PortfolioAllocationEnv, PortfolioEnvConfig


def _make_env_one_step(*, n_assets: int = 3) -> PortfolioAllocationEnv:
    # Two timesteps so we can step once.
    returns = [[0.01 for _ in range(n_assets)], [0.0 for _ in range(n_assets)]]
    volatility = [[0.0 for _ in range(n_assets)], [0.0 for _ in range(n_assets)]]
    covariance = [
        [[0.0 for _ in range(n_assets)] for _ in range(n_assets)],
        [[0.0 for _ in range(n_assets)] for _ in range(n_assets)],
    ]
    assets = [f"A{i}" for i in range(n_assets)]
    cfg = PortfolioEnvConfig(
        initial_cash=100.0,
        min_weight=0.0,
        max_weight=1.0,
        max_gross_exposure=1.0,
        transaction_cost_rate=0.001,
        slippage_rate=0.0,
        turnover_penalty=0.0,
    )
    return PortfolioAllocationEnv(
        returns=returns,
        volatility=volatility,
        covariance=covariance,
        assets=assets,
        config=cfg,
    )


def test_constraints_clip_and_normalize():
    env = _make_env_one_step(n_assets=3)
    env.reset()

    # Out-of-bounds action -> should be clipped to [0,1] then normalized to gross exposure=1.
    obs, reward, done, info = env.step([2.0, -1.0, 0.5])
    w = obs["portfolio"]["weights"]
    assert len(w) == 3
    assert all(0.0 <= x <= 1.0 for x in w)
    assert abs(sum(abs(x) for x in w) - 1.0) < 1e-12


def test_reward_matches_return_minus_costs():
    env = _make_env_one_step(n_assets=2)
    env.reset()

    # Move from all-zero to [0.5, 0.5] => turnover = 1.0
    obs, reward, done, info = env.step([0.5, 0.5])
    assert abs(info["turnover"] - 1.0) < 1e-12

    expected_portfolio_return = 0.01 * 0.5 + 0.01 * 0.5  # 0.01
    expected_cost = 1.0 * env.config.transaction_cost_rate
    expected_reward = expected_portfolio_return - expected_cost

    assert abs(info["portfolio_return"] - expected_portfolio_return) < 1e-12
    assert abs(info["cost"] - expected_cost) < 1e-12
    assert abs(reward - expected_reward) < 1e-12


def test_equity_updates_with_frictions():
    env = _make_env_one_step(n_assets=2)
    env.reset()

    obs, reward, done, info = env.step([1.0, 0.0])
    eq0 = info["equity_before"]
    eq1 = info["equity_after"]

    # equity_after = equity_before*(1+r) - equity_before*(cost+slippage)
    expected = eq0 * (1.0 + info["portfolio_return"]) - eq0 * (info["cost"] + info["slippage"])
    assert abs(eq1 - expected) < 1e-12
    assert eq1 <= eq0 * (1.0 + info["portfolio_return"]) + 1e-12

