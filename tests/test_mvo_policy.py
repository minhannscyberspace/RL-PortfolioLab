from __future__ import annotations

from rl_portfoliolab.benchmarks.policies import mean_variance_weights


def test_mean_variance_weights_basic_long_only():
    # Two assets: asset0 higher mean, low correlation -> weight should tilt to asset0.
    returns_window = [
        [0.02, 0.00],
        [0.02, 0.00],
        [0.02, 0.00],
    ]
    cov = [
        [0.01, 0.0],
        [0.0, 0.01],
    ]

    w = mean_variance_weights(returns_window=returns_window, cov_row=cov, ridge=1e-6)
    assert len(w) == 2
    assert abs(sum(w) - 1.0) < 1e-12
    assert w[0] > w[1]

