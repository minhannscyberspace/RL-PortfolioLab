from __future__ import annotations

from rl_portfoliolab.stress.regimes import classify_regimes


def test_classify_regimes_shapes():
    rets = [0.0] * 100
    masks = classify_regimes(rets, vol_window=10, high_vol_quantile=0.8, crash_quantile=0.05)
    assert len(masks.high_vol) == 100
    assert len(masks.low_vol) == 100
    assert len(masks.crash) == 100
    assert len(masks.calm) == 100


def test_crash_mask_triggers_on_worst_returns():
    rets = [0.01] * 50 + [-0.5] + [0.01] * 49
    masks = classify_regimes(rets, vol_window=5, high_vol_quantile=0.9, crash_quantile=0.05)
    assert masks.crash[50] is True

