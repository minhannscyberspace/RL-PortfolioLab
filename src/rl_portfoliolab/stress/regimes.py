from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


def _clean(xs: list[Optional[float]]) -> list[float]:
    out: list[float] = []
    for x in xs:
        if x is None:
            continue
        xf = float(x)
        if math.isnan(xf):
            continue
        out.append(xf)
    return out


def quantile(xs: list[Optional[float]], q: float) -> float:
    ys = _clean(xs)
    if not ys:
        return float("nan")
    if q <= 0:
        return min(ys)
    if q >= 1:
        return max(ys)
    ys.sort()
    idx = int(math.floor(q * (len(ys) - 1)))
    return ys[idx]


def rolling_std(xs: list[Optional[float]], window: int) -> list[Optional[float]]:
    ys = xs
    out: list[Optional[float]] = [None for _ in ys]
    if window < 2:
        return out
    for t in range(len(ys)):
        if t + 1 < window:
            continue
        w = ys[t - window + 1 : t + 1]
        w_clean = _clean(w)
        if len(w_clean) < 2:
            out[t] = None
            continue
        mu = sum(w_clean) / len(w_clean)
        var = sum((v - mu) ** 2 for v in w_clean) / (len(w_clean) - 1)
        out[t] = math.sqrt(var)
    return out


@dataclass(frozen=True)
class RegimeMasks:
    high_vol: list[bool]
    low_vol: list[bool]
    crash: list[bool]
    calm: list[bool]


def classify_regimes(
    portfolio_returns: list[Optional[float]],
    *,
    vol_window: int,
    high_vol_quantile: float,
    crash_quantile: float,
) -> RegimeMasks:
    """
    Create boolean masks over time steps.

    - high_vol/low_vol based on rolling std threshold
    - crash based on return <= crash threshold (quantile of returns)
    - calm = low_vol AND not crash
    """
    n = len(portfolio_returns)
    vol = rolling_std(portfolio_returns, window=vol_window)
    vol_thr = quantile(vol, high_vol_quantile)
    crash_thr = quantile(portfolio_returns, crash_quantile)

    high_vol = [False] * n
    low_vol = [False] * n
    crash = [False] * n
    calm = [False] * n

    for t in range(n):
        r = portfolio_returns[t]
        v = vol[t]

        if r is not None and not math.isnan(float(r)) and float(r) <= crash_thr:
            crash[t] = True

        if v is not None and not math.isnan(float(v)):
            if float(v) >= vol_thr:
                high_vol[t] = True
            else:
                low_vol[t] = True

        if low_vol[t] and not crash[t]:
            calm[t] = True

    return RegimeMasks(high_vol=high_vol, low_vol=low_vol, crash=crash, calm=calm)


def apply_mask(xs: list[Optional[float]], mask: list[bool]) -> list[Optional[float]]:
    if len(xs) != len(mask):
        raise ValueError("mask length must match series length")
    return [x for x, m in zip(xs, mask) if m]

