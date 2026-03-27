from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _is_nan(x: float) -> bool:
    return isinstance(x, float) and math.isnan(x)


def _clip(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _sum_abs(xs: list[float]) -> float:
    return sum(abs(x) for x in xs)


def _safe_normalize_weights(weights: list[float], *, max_gross_exposure: float) -> list[float]:
    """
    Normalize weights to satisfy gross exposure constraint:
    - if all zeros -> keep zeros
    - else scale so sum(abs(w)) == max_gross_exposure
    """
    gross = _sum_abs(weights)
    if gross == 0.0:
        return weights
    scale = max_gross_exposure / gross
    return [w * scale for w in weights]


@dataclass(frozen=True)
class PortfolioEnvConfig:
    initial_cash: float
    min_weight: float
    max_weight: float
    max_gross_exposure: float
    transaction_cost_rate: float
    slippage_rate: float
    turnover_penalty: float


@dataclass
class PortfolioState:
    t: int
    weights: list[float]
    cash: float
    equity: float


class PortfolioAllocationEnv:
    """
    Minimal Gym-style environment for portfolio allocation.

    - Observations: a dict with feature slices + current portfolio state.
    - Actions: target portfolio weights (list[float], length N).
    - Reward: portfolio return - transaction costs - slippage - turnover penalty.

    This env is intentionally dependency-light (no `gym` import) to keep the MVP runnable.
    """

    def __init__(
        self,
        *,
        returns: list[list[Optional[float]]],
        volatility: list[list[Optional[float]]],
        covariance: list[list[list[Optional[float]]]],
        assets: list[str],
        config: PortfolioEnvConfig,
    ) -> None:
        self.assets = list(assets)
        self.n_assets = len(self.assets)
        self.returns = returns
        self.volatility = volatility
        self.covariance = covariance
        self.config = config

        t_len = len(self.returns)
        if t_len == 0:
            raise ValueError("returns is empty.")
        if len(self.volatility) != t_len or len(self.covariance) != t_len:
            raise ValueError("returns/volatility/covariance must have the same time length.")

        self.state: Optional[PortfolioState] = None

    def reset(self) -> dict[str, Any]:
        w0 = [0.0 for _ in range(self.n_assets)]
        self.state = PortfolioState(
            t=0,
            weights=w0,
            cash=self.config.initial_cash,
            equity=self.config.initial_cash,
        )
        return self._observe()

    def step(self, action: list[float]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("Call reset() before step().")
        if len(action) != self.n_assets:
            raise ValueError(f"Action length {len(action)} does not match n_assets {self.n_assets}.")

        t = self.state.t
        if t >= len(self.returns):
            raise RuntimeError("Episode already finished.")

        prev_w = list(self.state.weights)
        target_w = self._apply_constraints(action)

        turnover = sum(abs(target_w[i] - prev_w[i]) for i in range(self.n_assets))
        cost = turnover * self.config.transaction_cost_rate
        slippage = turnover * self.config.slippage_rate
        turnover_pen = turnover * self.config.turnover_penalty

        # Portfolio return at time t from returns[t] (feature index aligned with prices).
        r_t = self._portfolio_return(t, target_w)

        # Apply equity update.
        prev_equity = self.state.equity
        gross_growth = 1.0 + r_t
        if gross_growth < 0.0:
            gross_growth = 0.0

        new_equity = prev_equity * gross_growth
        friction = (cost + slippage) * prev_equity
        new_equity = max(0.0, new_equity - friction)

        reward = r_t - cost - slippage - turnover_pen

        # Advance time.
        done = (t + 1) >= len(self.returns)
        self.state = PortfolioState(
            t=t + 1,
            weights=target_w,
            cash=self.state.cash,  # cash modeling deferred; treated implicitly via weights
            equity=new_equity,
        )

        info = {
            "t": t,
            "turnover": turnover,
            "cost": cost,
            "slippage": slippage,
            "turnover_penalty": turnover_pen,
            "portfolio_return": r_t,
            "equity_before": prev_equity,
            "equity_after": new_equity,
        }
        return self._observe(), reward, done, info

    def _apply_constraints(self, action: list[float]) -> list[float]:
        # Clip per-asset weights.
        clipped = [_clip(float(w), self.config.min_weight, self.config.max_weight) for w in action]

        # Enforce gross exposure.
        clipped = _safe_normalize_weights(clipped, max_gross_exposure=self.config.max_gross_exposure)

        # Final safety clip.
        clipped = [_clip(w, self.config.min_weight, self.config.max_weight) for w in clipped]
        return clipped

    def _portfolio_return(self, t: int, weights: list[float]) -> float:
        r_vec = self.returns[t]
        out = 0.0
        for j in range(self.n_assets):
            r = r_vec[j]
            if r is None or _is_nan(float(r)):
                continue
            out += weights[j] * float(r)
        return out

    def _observe(self) -> dict[str, Any]:
        if self.state is None:
            raise RuntimeError("Environment not reset.")
        t = self.state.t
        # Clamp t for observation at terminal step (return last features).
        if t >= len(self.returns):
            t_obs = len(self.returns) - 1
        else:
            t_obs = t

        return {
            "t": t,
            "assets": self.assets,
            "features": {
                "returns": self.returns[t_obs],
                "volatility": self.volatility[t_obs],
                "covariance": self.covariance[t_obs],
            },
            "portfolio": {
                "weights": list(self.state.weights),
                "equity": self.state.equity,
            },
        }

