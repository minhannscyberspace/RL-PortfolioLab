# RL-PortfolioLab — Baseline Report

## Quick navigation

- Dataset & provenance
- Methodology
- Metrics glossary
- Key takeaways
- Benchmark comparison
- Equity curves
- Trading frictions diagnostics
- Walk-forward robustness + worst folds
- Regime / stress breakdown

## Dataset & provenance

- **source**: stooq
- **tickers**: ['spy.us', 'qqq.us', 'iwm.us', 'tlt.us']
- **start_date**: 2005-01-01
- **end_date**: None
- **output_csv**: `data/raw/prices_wide.csv`

## Methodology (what was computed)

- **Data**: Daily close prices downloaded from Stooq into a wide CSV, then converted into returns + rolling stats.
- **Environment**: At each step t, the policy outputs target weights w_t.
- **Turnover**: sum_i |w_{t,i} - w_{t-1,i}|.
- **Costs**: `transaction_cost_rate * turnover` plus `slippage_rate * turnover` (applied to equity).
- **Reward**: approximately `portfolio_return - cost - slippage - turnover_penalty` (per-step).
- **Metrics**: computed over the episode and also within regimes (high_vol/low_vol/crash/calm).
- **Walk-forward**: repeated out-of-sample folds with expanding training window; reported Sharpe distribution across folds.

## Metrics glossary (how to read the tables)

- **final_equity**: ending portfolio value after the episode. **Higher is better**.
- **total_reward**: sum of per-step rewards. In this MVP env, reward is approximately: `portfolio_return - transaction_cost - slippage - turnover_penalty`. **Higher is better**.
- **sharpe**: mean(returns)/std(returns) over steps. **Higher is better**. Can be NaN if std is ~0.
- **max_drawdown**: worst peak→trough equity drop fraction (e.g. 0.25 = -25%). **Lower is better**.
- **cvar_5**: average of the worst 5% returns (tail risk). **Less negative / higher is better**.
- **avg_turnover**: average Σ|w_t − w_{t−1}|. High turnover implies high trading/costs. **Lower is better**.
- **walk-forward (folds/mean/median/p10/p90 sharpe)**: out-of-sample Sharpe distribution across folds. Prefer higher median and a less-negative p10 (worst-case periods).
- **regimes (high_vol/low_vol/crash/calm)**: metrics computed only on those slices. Use to see fragility.

## Key takeaways

- Best final equity: `equal_weight` (636401.6731233619)
- Best Sharpe (in-sample backtest): `inverse_vol` (0.05064722299243669)
## Benchmark comparison

_Interpretation_: compare `final_equity` and risk metrics (`max_drawdown`, `cvar_5`). A strategy can look great on equity but still be fragile in tail risk.
| policy | final_equity | total_reward | sharpe | max_drawdown | cvar_5 | avg_turnover |
| --- | --- | --- | --- | --- | --- | --- |
| equal_weight | 636402 | 2.08363 | 0.0418389 | 0.416946 | -0.0222086 | 0 |
| momentum | 79419.4 | -0.10965 | 0.0410666 | 0.530241 | -0.0232756 | 0.266747 |
| inverse_vol | 553955 | 1.85728 | 0.0506472 | 0.324683 | -0.0180707 | 0.0269394 |
| mean_variance | 164919 | 0.71096 | 0.0378615 | 0.496459 | -0.0257457 | 0.16692 |

## Trading frictions diagnostics (from episode series)

_Interpretation_: these are **rate terms** per-step aggregated over the episode (not dollar costs). Higher turnover/costs indicate less realistic trading behavior._
| policy | steps | avg_turnover | total_cost_rate | total_slippage_rate | total_turnover_penalty |
| --- | --- | --- | --- | --- | --- |
| equal_weight | 5340 | 0.000187266 | 0.001 | 0.0005 | 0.0001 |
| momentum | 5340 | 0.266884 | 1.42516 | 0.71258 | 0.142516 |
| inverse_vol | 5340 | 0.0271216 | 0.144829 | 0.0724147 | 0.0144829 |
| mean_variance | 5340 | 0.167076 | 0.892185 | 0.446092 | 0.0892185 |

## Red flags (automatic heuristics)

_Thresholds used_: crash_sharpe>=-0.5, max_drawdown<=0.45, cvar_5>=-0.03, avg_turnover<=0.2, total_cost_rate<=1.0, pct_pos_sharpe>=55.0%, p10_sharpe>=-0.5.

### 1) Regime fragility (crash behavior)

- `equal_weight` crash Sharpe is -2.33 (threshold: >= -0.5).
- `inverse_vol` crash Sharpe is -2.58 (threshold: >= -0.5).
- `mean_variance` crash Sharpe is -2.21 (threshold: >= -0.5).
- `momentum` crash Sharpe is -2.64 (threshold: >= -0.5).

### 2) Drawdown & tail-risk warnings

- `momentum` max_drawdown is 0.53 (threshold: <= 0.45).
- `mean_variance` max_drawdown is 0.496 (threshold: <= 0.45).

### 3) Turnover & trading-cost warnings

- `momentum` avg_turnover is 0.267 (threshold: <= 0.2).
- `momentum` total_cost_rate is 1.43 (threshold: <= 1.0).

### 4) Walk-forward stability warnings

_(none triggered)_

### 5) Summary

- Red flags triggered.

## Walk-forward robustness (Sharpe across folds)

_Interpretation_: higher `median_sharpe` is good; very negative `p10_sharpe` indicates instability in some periods._
| policy | folds | pct_pos_sharpe | mean_sharpe | median_sharpe | min_sharpe | p10_sharpe | p90_sharpe | worst_fold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| equal_weight | 176 | 67.0455 | 0.0893115 | 0.0859273 | -0.31124 | -0.148562 | 0.315986 | 23 |
| inverse_vol | 176 | 65.9091 | 0.0956106 | 0.0880787 | -0.322511 | -0.152635 | 0.335956 | 29 |
| mean_variance | 176 | 65.3409 | 0.0897208 | 0.0911239 | -0.270672 | -0.144858 | 0.305879 | 23 |
| momentum | 176 | 63.0682 | 0.0652019 | 0.0812615 | -0.468837 | -0.179589 | 0.290253 | 140 |

## Walk-forward worst folds (lowest Sharpe)

_Interpretation_: these are the periods where a strategy performed worst out-of-sample. If these are extremely negative, the strategy is unstable._
| policy | fold | sharpe | final_equity | test_start_date | test_end_date |
| --- | --- | --- | --- | --- | --- |
| equal_weight | 23 | -0.31124 | 90507.1 | 2007-12-27T00:00:00 | 2008-02-08T00:00:00 |
| equal_weight | 113 | -0.293517 | 91267.6 | 2018-09-18T00:00:00 | 2018-10-29T00:00:00 |
| equal_weight | 29 | -0.280631 | 75378.5 | 2008-09-15T00:00:00 | 2008-10-24T00:00:00 |
| equal_weight | 32 | -0.266318 | 87530.6 | 2009-01-23T00:00:00 | 2009-03-06T00:00:00 |
| equal_weight | 63 | -0.258443 | 95912.9 | 2012-10-01T00:00:00 | 2012-11-13T00:00:00 |
| momentum | 140 | -0.468837 | 85409.9 | 2021-12-06T00:00:00 | 2022-01-18T00:00:00 |
| momentum | 175 | -0.342448 | 89729.3 | 2026-02-12T00:00:00 | 2026-03-26T00:00:00 |
| momentum | 121 | -0.300598 | 92170.8 | 2019-09-03T00:00:00 | 2019-10-14T00:00:00 |
| momentum | 23 | -0.290436 | 88222.7 | 2007-12-27T00:00:00 | 2008-02-08T00:00:00 |
| momentum | 32 | -0.258812 | 83693.8 | 2009-01-23T00:00:00 | 2009-03-06T00:00:00 |
| inverse_vol | 29 | -0.322511 | 81671.2 | 2008-09-15T00:00:00 | 2008-10-24T00:00:00 |
| inverse_vol | 113 | -0.316621 | 92387.9 | 2018-09-18T00:00:00 | 2018-10-29T00:00:00 |
| inverse_vol | 23 | -0.31105 | 91782.4 | 2007-12-27T00:00:00 | 2008-02-08T00:00:00 |
| inverse_vol | 32 | -0.289816 | 88323.8 | 2009-01-23T00:00:00 | 2009-03-06T00:00:00 |
| inverse_vol | 63 | -0.251795 | 95889.2 | 2012-10-01T00:00:00 | 2012-11-13T00:00:00 |
| mean_variance | 23 | -0.270672 | 90919.2 | 2007-12-27T00:00:00 | 2008-02-08T00:00:00 |
| mean_variance | 32 | -0.261499 | 85988.8 | 2009-01-23T00:00:00 | 2009-03-06T00:00:00 |
| mean_variance | 175 | -0.237032 | 90803.5 | 2026-02-12T00:00:00 | 2026-03-26T00:00:00 |
| mean_variance | 113 | -0.229199 | 89298.9 | 2018-09-18T00:00:00 | 2018-10-29T00:00:00 |
| mean_variance | 63 | -0.226026 | 94032 | 2012-10-01T00:00:00 | 2012-11-13T00:00:00 |

## Regime / stress breakdown

_Interpretation_: focus on `crash` and `high_vol`. If performance collapses there, the strategy is not robust._
| policy | regime | steps | sharpe | max_drawdown | cvar_5 | final_equity |
| --- | --- | --- | --- | --- | --- | --- |
| equal_weight | high_vol | 1065 | 0.0261145 | 0.404704 | -0.0366747 | 581985 |
| equal_weight | low_vol | 4256 | 0.0584359 | 0.276816 | -0.0155736 | 636402 |
| equal_weight | crash | 267 | -2.32978 | 0.4063 | -0.0532659 | 636402 |
| equal_weight | calm | 4142 | 0.152432 | 0.276816 | -0.0112499 | 647346 |
| inverse_vol | high_vol | 1065 | 0.0156244 | 0.283599 | -0.0282542 | 553955 |
| inverse_vol | low_vol | 4256 | 0.0757778 | 0.22813 | -0.0133559 | 563097 |
| inverse_vol | crash | 267 | -2.57614 | 0.309695 | -0.0413216 | 553955 |
| inverse_vol | calm | 4146 | 0.163981 | 0.22813 | -0.0100019 | 563097 |
| mean_variance | high_vol | 1065 | 0.0157171 | 0.468516 | -0.043471 | 164919 |
| mean_variance | low_vol | 4256 | 0.0612454 | 0.418882 | -0.0168628 | 168146 |
| mean_variance | crash | 267 | -2.20936 | 0.46113 | -0.0624632 | 164919 |
| mean_variance | calm | 4160 | 0.14416 | 0.418882 | -0.0127398 | 168146 |
| momentum | high_vol | 1065 | 0.035325 | 0.512993 | -0.0367264 | 86770.8 |
| momentum | low_vol | 4256 | 0.0495628 | 0.528387 | -0.017162 | 79419.4 |
| momentum | crash | 267 | -2.636 | 0.51171 | -0.050941 | 79419.4 |
| momentum | calm | 4141 | 0.140189 | 0.528387 | -0.0125732 | 80785.2 |

## RL policy (SB3) evaluation

Summary:

```json
{
  "steps": 119,
  "total_reward": 0.09578082864455834,
  "final_equity": 109848.95203876785,
  "avg_turnover": 0.008403361344537815,
  "avg_cost": 8.403361344537814e-06
}
```