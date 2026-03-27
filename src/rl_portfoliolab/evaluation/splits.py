from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WalkForwardSplit:
    train_start: int
    train_end: int  # exclusive
    test_start: int
    test_end: int  # exclusive


def make_walkforward_splits(
    *,
    n: int,
    train_size: int,
    test_size: int,
    step_size: int,
) -> list[WalkForwardSplit]:
    """
    Generate expanding walk-forward splits over [0, n).

    First split:
      train: [0, train_size)
      test:  [train_size, train_size + test_size)

    Next split advances by step_size:
      train: [0, train_size + step_size)
      test:  [train_size + step_size, train_size + step_size + test_size)
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    if train_size <= 0 or test_size <= 0 or step_size <= 0:
        raise ValueError("train_size/test_size/step_size must be > 0")

    splits: list[WalkForwardSplit] = []
    k = 0
    while True:
        train_end = train_size + k * step_size
        test_start = train_end
        test_end = test_start + test_size
        if test_end > n:
            break
        splits.append(
            WalkForwardSplit(
                train_start=0,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        k += 1
    if not splits:
        raise ValueError("No valid splits for given n/train_size/test_size/step_size")
    return splits

