from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional


def _clean(xs: Iterable[Optional[float]]) -> list[float]:
    out: list[float] = []
    for x in xs:
        if x is None:
            continue
        xf = float(x)
        if math.isnan(xf):
            continue
        out.append(xf)
    return out


def max_drawdown(equity: list[Optional[float]]) -> float:
    xs = _clean(equity)
    if not xs:
        return float("nan")
    peak = xs[0]
    mdd = 0.0
    for v in xs:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak != 0 else 0.0
        if dd > mdd:
            mdd = dd
    return mdd


def mean(xs: list[Optional[float]]) -> float:
    ys = _clean(xs)
    if not ys:
        return float("nan")
    return sum(ys) / len(ys)


def std(xs: list[Optional[float]]) -> float:
    ys = _clean(xs)
    if len(ys) < 2:
        return float("nan")
    mu = sum(ys) / len(ys)
    var = sum((y - mu) ** 2 for y in ys) / (len(ys) - 1)
    return math.sqrt(var)


def sharpe(returns: list[Optional[float]], *, eps: float = 1e-12) -> float:
    mu = mean(returns)
    sd = std(returns)
    if math.isnan(mu) or math.isnan(sd) or sd < eps:
        return float("nan")
    return mu / sd


def cvar(returns: list[Optional[float]], *, alpha: float = 0.05) -> float:
    ys = _clean(returns)
    if not ys:
        return float("nan")
    ys_sorted = sorted(ys)
    k = max(1, int(math.ceil(alpha * len(ys_sorted))))
    tail = ys_sorted[:k]
    return sum(tail) / len(tail)


def turnover_from_weights(weights: list[list[Optional[float]]]) -> float:
    # Average L1 turnover between consecutive weight vectors.
    if len(weights) < 2:
        return 0.0
    n = len(weights[0])
    total = 0.0
    steps = 0
    for t in range(1, len(weights)):
        w0 = weights[t - 1]
        w1 = weights[t]
        if len(w0) != n or len(w1) != n:
            raise ValueError("Inconsistent weight vector length.")
        tv = 0.0
        for j in range(n):
            a = w0[j]
            b = w1[j]
            if a is None or b is None:
                continue
            af = float(a)
            bf = float(b)
            if math.isnan(af) or math.isnan(bf):
                continue
            tv += abs(bf - af)
        total += tv
        steps += 1
    return total / steps if steps > 0 else 0.0

