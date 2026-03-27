from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

from rl_portfoliolab.data.types import WideMarketData


@dataclass(frozen=True)
class FeatureArrays:
    """
    Pure-Python feature arrays aligned to a common time index.
    """

    returns: list[list[float]]  # (T, N)
    volatility: list[list[float]]  # (T, N)
    covariance: list[list[list[float]]]  # (T, N, N)
    dates: list[object]  # (T,) datetime objects
    assets: list[str]


def build_features_from_prices(
    market_data: WideMarketData,
    *,
    return_lag: int,
    vol_window: int,
    cov_window: int,
    min_periods: int,
) -> FeatureArrays:
    """
    Build RL-ready state features using aligned prices:
    - returns (pct_change)
    - volatility (rolling std of returns)
    - covariance (rolling covariance matrix of returns)
    """
    if not market_data.dates:
        raise ValueError("market_data has no dates.")
    if len(market_data.prices) != len(market_data.dates):
        raise ValueError("market_data.prices length must match market_data.dates length.")

    if return_lag < 1:
        raise ValueError("return_lag must be >= 1")
    if vol_window < 2:
        raise ValueError("vol_window must be >= 2")
    if cov_window < 2:
        raise ValueError("cov_window must be >= 2")
    if min_periods < 1:
        raise ValueError("min_periods must be >= 1")

    n_assets = len(market_data.assets)
    t_all = len(market_data.dates)
    if t_all <= return_lag:
        raise ValueError("Not enough time points to compute returns with the given return_lag.")

    def nan() -> float:
        return float("nan")

    # Returns:
    # returns[t] corresponds to prices[t] and prices[t-return_lag] (after aligning to the return index).
    # We drop the first `return_lag` time steps, so T = t_all - return_lag.
    t = t_all - return_lag
    dates: list[object] = list(market_data.dates[return_lag:])

    returns: list[list[float]] = []
    for t_idx in range(return_lag, t_all):
        row_t = market_data.prices[t_idx]
        row_lag = market_data.prices[t_idx - return_lag]
        out_row: list[float] = []
        for j in range(n_assets):
            p_now = row_t[j]
            p_prev = row_lag[j]
            if math.isnan(p_now) or math.isnan(p_prev) or p_prev == 0.0:
                out_row.append(nan())
            else:
                out_row.append(p_now / p_prev - 1.0)
        returns.append(out_row)

    # Volatility (rolling std over returns, using ddof=0).
    volatility: list[list[float]] = [[nan() for _ in range(n_assets)] for _ in range(t)]
    for t_idx in range(t):
        if t_idx + 1 < vol_window:
            continue
        window_slice = returns[t_idx - vol_window + 1 : t_idx + 1]  # len = vol_window

        for j in range(n_assets):
            col = [row[j] for row in window_slice]
            if any(math.isnan(x) for x in col):
                volatility[t_idx][j] = nan()
                continue

            if vol_window < min_periods:
                volatility[t_idx][j] = nan()
                continue

            mu = sum(col) / len(col)
            var = sum((x - mu) ** 2 for x in col) / len(col)
            volatility[t_idx][j] = math.sqrt(var)

    # Covariance matrices (rolling covariance over return vectors).
    # Output covariance[t_idx] is an N x N matrix.
    covariance: list[list[list[float]]] = [
        [[nan() for _ in range(n_assets)] for _ in range(n_assets)] for _ in range(t)
    ]
    for t_idx in range(t):
        if t_idx + 1 < cov_window:
            continue
        window_slice = returns[t_idx - cov_window + 1 : t_idx + 1]  # len = cov_window

        # Strict behavior: if any NaN exists in the window, emit NaNs for the whole matrix.
        if any(any(math.isnan(x) for x in row) for row in window_slice):
            continue
        if cov_window < min_periods:
            continue

        means = []
        for j in range(n_assets):
            col = [row[j] for row in window_slice]
            means.append(sum(col) / len(col))

        m = len(window_slice)
        denom = m - 1
        if denom <= 0:
            continue

        cov_mat: list[list[float]] = [[nan() for _ in range(n_assets)] for _ in range(n_assets)]
        for i in range(n_assets):
            for j in range(n_assets):
                vi = [row[i] - means[i] for row in window_slice]
                vj = [row[j] - means[j] for row in window_slice]
                cov = sum(vi_k * vj_k for vi_k, vj_k in zip(vi, vj)) / denom
                cov_mat[i][j] = cov

        covariance[t_idx] = cov_mat

    return FeatureArrays(
        returns=returns,
        volatility=volatility,
        covariance=covariance,
        dates=dates,
        assets=list(market_data.assets),
    )

