from __future__ import annotations

from rl_portfoliolab.evaluation.metrics import cvar, max_drawdown, sharpe, turnover_from_weights


def test_max_drawdown_basic():
    equity = [100.0, 110.0, 105.0, 120.0, 90.0]
    # Peak=120 -> trough=90 => 25% drawdown
    assert abs(max_drawdown(equity) - 0.25) < 1e-12


def test_sharpe_simple():
    rets = [0.01, 0.01, 0.01, 0.01]
    # zero variance -> sharpe NaN
    s = sharpe(rets)
    assert s != s  # NaN check


def test_cvar_tail_mean():
    rets = [0.1, 0.0, -0.2, -0.1, 0.05]
    # alpha=0.4 => k=ceil(0.4*5)=2 => average of two worst: (-0.2 + -0.1)/2 = -0.15
    assert abs(cvar(rets, alpha=0.4) - (-0.15)) < 1e-12


def test_turnover_from_weights():
    ws = [
        [0.5, 0.5],
        [0.6, 0.4],
        [0.4, 0.6],
    ]
    # turnover steps: |0.6-0.5|+|0.4-0.5|=0.2; then |0.4-0.6|+|0.6-0.4|=0.4 => avg 0.3
    assert abs(turnover_from_weights(ws) - 0.3) < 1e-12

