from __future__ import annotations

from rl_portfoliolab.evaluation.splits import make_walkforward_splits


def test_make_walkforward_splits_counts_and_bounds():
    splits = make_walkforward_splits(n=200, train_size=60, test_size=30, step_size=30)
    assert len(splits) == 4
    for s in splits:
        assert s.train_start == 0
        assert s.train_end == s.test_start
        assert s.test_end > s.test_start
        assert s.test_end <= 200

