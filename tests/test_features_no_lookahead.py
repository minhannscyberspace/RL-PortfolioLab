from __future__ import annotations

import math
import random
from datetime import datetime, timedelta

from rl_portfoliolab.data.alignment import align_price_matrix
from rl_portfoliolab.features.builder import build_features_from_prices
from rl_portfoliolab.data.types import WideMarketData
from rl_portfoliolab.utils.seeding import set_seed


def _make_synthetic_wide_market_data(*, seed: int, n_days: int, n_assets: int) -> WideMarketData:
    set_seed(seed)
    assets = [f"A{i}" for i in range(n_assets)]
    start = datetime(2020, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n_days)]

    log_price = [math.log(random.uniform(50, 150)) for _ in assets]
    prices: list[list[float]] = []
    for _ in range(n_days):
        row: list[float] = []
        for j in range(n_assets):
            log_price[j] += random.gauss(0.0, 0.01)
            row.append(math.exp(log_price[j]))
        prices.append(row)

    return WideMarketData(dates=dates, assets=assets, prices=prices)


def _assert_close(a: float, b: float, *, atol: float = 1e-12) -> None:
    if math.isnan(a) and math.isnan(b):
        return
    assert abs(a - b) <= atol, f"Values differ: {a} vs {b}"


def test_no_lookahead_in_features_for_future_changes():
    # Parameters chosen to exercise rolling computations.
    return_lag = 1
    vol_window = 5
    cov_window = 10
    min_periods = 3

    n_days = 80
    n_assets = 3
    market_data = _make_synthetic_wide_market_data(seed=123, n_days=n_days, n_assets=n_assets)
    aligned = align_price_matrix(market_data, missing_value_strategy="ffill")

    features1 = build_features_from_prices(
        aligned,
        return_lag=return_lag,
        vol_window=vol_window,
        cov_window=cov_window,
        min_periods=min_periods,
    )

    # Modify future prices only (beyond `cutoff_price_index`).
    cutoff_price_index = 60
    future_start = cutoff_price_index + 1
    prices2 = [row[:] for row in aligned.prices]
    for i in range(future_start, len(prices2)):
        for j in range(n_assets):
            prices2[i][j] = prices2[i][j] * 10.0  # big change, still valid prices
    market_data2 = WideMarketData(dates=aligned.dates, assets=aligned.assets, prices=prices2)

    features2 = build_features_from_prices(
        market_data2,
        return_lag=return_lag,
        vol_window=vol_window,
        cov_window=cov_window,
        min_periods=min_periods,
    )

    # For return_lag=1, return feature index k depends on prices[k] and prices[k+1].
    # If we require k+1 <= cutoff_price_index, then k <= cutoff_price_index - 1.
    k_max = cutoff_price_index - 1

    for t in range(k_max + 1):
        for j in range(n_assets):
            _assert_close(features1.returns[t][j], features2.returns[t][j])
            _assert_close(features1.volatility[t][j], features2.volatility[t][j])
        for i in range(n_assets):
            for j in range(n_assets):
                _assert_close(features1.covariance[t][i][j], features2.covariance[t][i][j])

