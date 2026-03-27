from __future__ import annotations

import math
from typing import Optional


def _is_valid(x: Optional[float]) -> bool:
    if x is None:
        return False
    xf = float(x)
    return not math.isnan(xf)


def equal_weight(n_assets: int) -> list[float]:
    if n_assets <= 0:
        raise ValueError("n_assets must be > 0")
    return [1.0 / n_assets for _ in range(n_assets)]


def momentum_weights(
    returns_window: list[list[Optional[float]]],
) -> list[float]:
    """
    Simple cross-sectional momentum:
    - compute mean return per asset over the window
    - keep only positive means, normalize to sum=1
    - if none positive, fall back to equal weight
    """
    if not returns_window:
        raise ValueError("returns_window is empty")
    n = len(returns_window[0])
    scores = [0.0 for _ in range(n)]
    counts = [0 for _ in range(n)]

    for row in returns_window:
        if len(row) != n:
            raise ValueError("Inconsistent returns row length")
        for j in range(n):
            r = row[j]
            if _is_valid(r):
                scores[j] += float(r)
                counts[j] += 1

    means = []
    for j in range(n):
        means.append(scores[j] / counts[j] if counts[j] > 0 else float("nan"))

    pos = [max(0.0, m) if not math.isnan(m) else 0.0 for m in means]
    s = sum(pos)
    if s <= 0.0:
        return equal_weight(n)
    return [p / s for p in pos]


def inverse_vol_weights(vol_row: list[Optional[float]]) -> list[float]:
    """
    Risk parity proxy: weights ∝ 1/vol
    """
    n = len(vol_row)
    inv = []
    for v in vol_row:
        if not _is_valid(v):
            inv.append(0.0)
        else:
            vf = float(v)
            inv.append(0.0 if vf <= 0 else 1.0 / vf)
    s = sum(inv)
    if s <= 0.0:
        return equal_weight(n)
    return [x / s for x in inv]


def _solve_linear_system(a: list[list[float]], b: list[float]) -> list[float]:
    """
    Solve Ax=b using Gaussian elimination with partial pivoting.
    Pure-Python to keep benchmarks runnable without NumPy.
    """
    n = len(a)
    if n == 0 or any(len(row) != n for row in a) or len(b) != n:
        raise ValueError("Invalid shapes for linear system")

    # Build augmented matrix.
    m = [row[:] + [b_i] for row, b_i in zip(a, b)]

    for col in range(n):
        # Pivot
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-18:
            raise ValueError("Singular matrix")
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]

        # Normalize pivot row
        piv = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= piv

        # Eliminate other rows
        for r in range(n):
            if r == col:
                continue
            factor = m[r][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                m[r][j] -= factor * m[col][j]

    return [m[i][n] for i in range(n)]


def mean_variance_weights(
    *,
    returns_window: list[list[Optional[float]]],
    cov_row: list[list[Optional[float]]],
    ridge: float = 1e-6,
) -> list[float]:
    """
    Mean-variance (MVO) proxy (long-only):
    - mu = mean returns over window
    - Sigma = covariance matrix (from Phase 1 rolling cov)
    - w ∝ Sigma^{-1} mu  (with ridge added to diagonal)
    - clip negative weights to 0, then normalize to sum=1
    """
    if not returns_window:
        raise ValueError("returns_window is empty")

    n = len(returns_window[0])
    if n == 0:
        raise ValueError("No assets")
    if any(len(r) != n for r in returns_window):
        raise ValueError("Inconsistent returns row length")
    if len(cov_row) != n or any(len(r) != n for r in cov_row):
        raise ValueError("Invalid covariance matrix shape")

    # mu
    sums = [0.0 for _ in range(n)]
    counts = [0 for _ in range(n)]
    for row in returns_window:
        for j in range(n):
            r = row[j]
            if _is_valid(r):
                sums[j] += float(r)
                counts[j] += 1
    mu = [sums[j] / counts[j] if counts[j] > 0 else 0.0 for j in range(n)]

    # Sigma with ridge
    sigma: list[list[float]] = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            v = cov_row[i][j]
            sigma[i][j] = float(v) if _is_valid(v) else 0.0
        sigma[i][i] += float(ridge)

    try:
        w_raw = _solve_linear_system(sigma, mu)
    except ValueError:
        return equal_weight(n)

    w_pos = [max(0.0, float(w)) for w in w_raw]
    s = sum(w_pos)
    if s <= 0.0:
        return equal_weight(n)
    return [w / s for w in w_pos]

