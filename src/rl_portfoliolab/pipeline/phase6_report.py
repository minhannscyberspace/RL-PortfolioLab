from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from rl_portfoliolab.reporting.report_templates import html_page, html_table, markdown_table, sparkline_svg
from rl_portfoliolab.utils.seeding import set_seed


@dataclass(frozen=True)
class Phase6Config:
    seed: int
    benchmarks_comparison_path: str
    stress_summary_path: str
    walkforward_summary_path: str
    rl_eval_path: Optional[str]
    data_config_path: Optional[str]
    phase2_env_config_path: Optional[str]
    out_dir: str
    run_name: str
    crash_sharpe_max: float = -0.5
    max_drawdown_max: float = 0.45
    cvar5_min: float = -0.03
    avg_turnover_max: float = 0.20
    total_cost_rate_max: float = 1.0
    pct_pos_sharpe_min: float = 55.0
    p10_sharpe_min: float = -0.5


def _require_mapping(x: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(x, Mapping):
        raise ValueError(f"`{name}` must be an object/mapping.")
    return x


def _require_str(x: Any, name: str) -> str:
    if not isinstance(x, str):
        raise ValueError(f"`{name}` must be a string.")
    return x


def _require_int(x: Any, name: str) -> int:
    if not isinstance(x, int):
        raise ValueError(f"`{name}` must be an int.")
    return x


def _get_float(mapping: Mapping[str, Any], key: str, default: float) -> float:
    v = mapping.get(key, default)
    return float(v) if isinstance(v, (int, float)) else default


def load_phase6_config(path: str | Path) -> Phase6Config:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    root = _require_mapping(raw, "root")
    inputs = _require_mapping(root.get("inputs"), "inputs")
    output = _require_mapping(root.get("output"), "output")
    red_flags_raw = root.get("red_flags")
    red_flags = red_flags_raw if isinstance(red_flags_raw, Mapping) else {}

    rl_eval = inputs.get("rl_eval_path")
    rl_eval_path = rl_eval if isinstance(rl_eval, str) else None
    data_cfg = inputs.get("data_config_path")
    data_config_path = data_cfg if isinstance(data_cfg, str) else None
    p2 = inputs.get("phase2_env_config_path")
    phase2_env_config_path = p2 if isinstance(p2, str) else None

    return Phase6Config(
        seed=_require_int(root.get("seed"), "seed"),
        benchmarks_comparison_path=_require_str(inputs.get("benchmarks_comparison_path"), "inputs.benchmarks_comparison_path"),
        stress_summary_path=_require_str(inputs.get("stress_summary_path"), "inputs.stress_summary_path"),
        walkforward_summary_path=_require_str(inputs.get("walkforward_summary_path"), "inputs.walkforward_summary_path"),
        rl_eval_path=rl_eval_path,
        data_config_path=data_config_path,
        phase2_env_config_path=phase2_env_config_path,
        out_dir=_require_str(output.get("out_dir"), "output.out_dir"),
        run_name=_require_str(output.get("run_name"), "output.run_name"),
        crash_sharpe_max=_get_float(red_flags, "crash_sharpe_max", -0.5),
        max_drawdown_max=_get_float(red_flags, "max_drawdown_max", 0.45),
        cvar5_min=_get_float(red_flags, "cvar5_min", -0.03),
        avg_turnover_max=_get_float(red_flags, "avg_turnover_max", 0.20),
        total_cost_rate_max=_get_float(red_flags, "total_cost_rate_max", 1.0),
        pct_pos_sharpe_min=_get_float(red_flags, "pct_pos_sharpe_min", 55.0),
        p10_sharpe_min=_get_float(red_flags, "p10_sharpe_min", -0.5),
    )


def _read_json(path: str | Path) -> Any:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def _safe_equity_series_from_eval(path: Path) -> list[float]:
    """
    Read an eval JSON file (Phase 4 or Phase 3) and extract equity_after series.
    """
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    series = obj.get("series")
    if not isinstance(series, list):
        return []
    out: list[float] = []
    for row in series:
        if not isinstance(row, Mapping):
            continue
        v = row.get("equity_after")
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


def _safe_float_series(eval_obj: Any, key: str) -> list[float]:
    if not isinstance(eval_obj, Mapping):
        return []
    series = eval_obj.get("series")
    if not isinstance(series, list):
        return []
    out: list[float] = []
    for row in series:
        if not isinstance(row, Mapping):
            continue
        v = row.get(key)
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


def _sum(xs: list[float]) -> float:
    return float(sum(xs))


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _median(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    mid = len(ys) // 2
    return ys[mid] if len(ys) % 2 == 1 else 0.5 * (ys[mid - 1] + ys[mid])


def _pctl(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    if q <= 0:
        return ys[0]
    if q >= 1:
        return ys[-1]
    idx = int((len(ys) - 1) * q)
    return ys[idx]


def _pct_positive(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    return 100.0 * (sum(1 for x in xs if x > 0) / len(xs))


@dataclass(frozen=True)
class RedFlagThresholds:
    crash_sharpe_max: float = -0.5
    max_drawdown_max: float = 0.45
    cvar5_min: float = -0.03
    avg_turnover_max: float = 0.20
    total_cost_rate_max: float = 1.0
    pct_pos_sharpe_min: float = 55.0
    p10_sharpe_min: float = -0.5


def run_phase6(cfg: Phase6Config) -> tuple[Path, Path]:
    set_seed(cfg.seed)

    comparison = _read_json(cfg.benchmarks_comparison_path)
    stress = _read_json(cfg.stress_summary_path)
    walkforward = _read_json(cfg.walkforward_summary_path)

    rl_eval = None
    if cfg.rl_eval_path is not None and Path(cfg.rl_eval_path).exists():
        rl_eval = _read_json(cfg.rl_eval_path)

    data_cfg = None
    if cfg.data_config_path is not None and Path(cfg.data_config_path).exists():
        # YAML config, keep as raw text snippet + parsed keys if possible
        try:
            import yaml as _yaml

            data_cfg = _yaml.safe_load(Path(cfg.data_config_path).read_text(encoding="utf-8"))
        except Exception:
            data_cfg = None

    # Optional: load Phase 1 feature dates so we can show date ranges for walk-forward folds.
    feature_dates: list[str] = []
    if cfg.phase2_env_config_path is not None and Path(cfg.phase2_env_config_path).exists():
        try:
            from rl_portfoliolab.pipeline.phase2_env import load_phase2_env_config, load_phase1_features_json

            p2cfg = load_phase2_env_config(cfg.phase2_env_config_path)
            p1raw = _read_json(p2cfg.phase1_features_path)
            if isinstance(p1raw, Mapping) and isinstance(p1raw.get("dates"), list):
                feature_dates = [str(x) for x in p1raw.get("dates")]
        except Exception:
            feature_dates = []

    out_dir = Path(cfg.out_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Benchmarks table
    bench_cols = ["policy", "final_equity", "total_reward", "sharpe", "max_drawdown", "cvar_5", "avg_turnover"]

    # Stress summary: flatten to one row per (policy, regime)
    stress_rows = []
    for entry in stress:
        policy = entry.get("policy")
        regimes = entry.get("regimes", {})
        for regime_name, metrics in regimes.items():
            stress_rows.append(
                {
                    "policy": policy,
                    "regime": regime_name,
                    "sharpe": metrics.get("sharpe"),
                    "max_drawdown": metrics.get("max_drawdown"),
                    "cvar_5": metrics.get("cvar_5"),
                    "final_equity": metrics.get("final_equity"),
                    "steps": metrics.get("steps"),
                }
            )
    stress_cols = ["policy", "regime", "steps", "sharpe", "max_drawdown", "cvar_5", "final_equity"]

    # Walk-forward aggregate summary (mean/median sharpe per policy, fold count)
    wf_rows = []
    folds = walkforward.get("folds") if isinstance(walkforward, Mapping) else None
    if isinstance(folds, list) and folds:
        # policy -> list of sharpe values across folds
        per_policy: dict[str, list[float]] = {}
        # policy -> list of (fold_idx, sharpe) for worst fold tracking
        per_policy_folded: dict[str, list[tuple[int, float]]] = {}
        for fold in folds:
            if not isinstance(fold, Mapping):
                continue
            results = fold.get("results")
            if not isinstance(results, list):
                continue
            fold_idx = fold.get("fold")
            fold_i = int(fold_idx) if isinstance(fold_idx, int) else -1
            for r in results:
                if not isinstance(r, Mapping):
                    continue
                policy = r.get("policy")
                summ = r.get("summary", {})
                if not isinstance(policy, str) or not isinstance(summ, Mapping):
                    continue
                s = summ.get("sharpe")
                if isinstance(s, (int, float)):
                    per_policy.setdefault(policy, []).append(float(s))
                    per_policy_folded.setdefault(policy, []).append((fold_i, float(s)))

        for policy, xs in sorted(per_policy.items()):
            worst_fold = None
            if policy in per_policy_folded and per_policy_folded[policy]:
                wf = min(per_policy_folded[policy], key=lambda t: t[1])
                worst_fold = wf[0]
            wf_rows.append(
                {
                    "policy": policy,
                    "folds": len(xs),
                    "pct_pos_sharpe": _pct_positive(xs),
                    "mean_sharpe": _mean(xs),
                    "median_sharpe": _median(xs),
                    "min_sharpe": min(xs) if xs else None,
                    "p10_sharpe": _pctl(xs, 0.10),
                    "p90_sharpe": _pctl(xs, 0.90),
                    "worst_fold": worst_fold,
                }
            )
    wf_cols = [
        "policy",
        "folds",
        "pct_pos_sharpe",
        "mean_sharpe",
        "median_sharpe",
        "min_sharpe",
        "p10_sharpe",
        "p90_sharpe",
        "worst_fold",
    ]

    # Key takeaways (simple ranking by final_equity and sharpe)
    takeaways = []
    if isinstance(comparison, list) and comparison:
        try:
            best_equity = max(comparison, key=lambda r: float(r.get("final_equity", float("-inf"))))
            best_sharpe = max(comparison, key=lambda r: float(r.get("sharpe", float("-inf"))))
            takeaways.append(f"- Best final equity: `{best_equity.get('policy')}` ({best_equity.get('final_equity')})")
            takeaways.append(f"- Best Sharpe (in-sample backtest): `{best_sharpe.get('policy')}` ({best_sharpe.get('sharpe')})")
        except Exception:
            pass

    md_parts = []
    md_parts.append("# RL-PortfolioLab — Baseline Report\n")
    md_parts.append("## Quick navigation\n")
    md_parts.append(
        "\n".join(
            [
                "- Dataset & provenance",
                "- Methodology",
                "- Metrics glossary",
                "- Key takeaways",
                "- Benchmark comparison",
                "- Equity curves",
                "- Trading frictions diagnostics",
                "- Walk-forward robustness + worst folds",
                "- Regime / stress breakdown",
            ]
        )
    )
    md_parts.append("")
    md_parts.append("## Dataset & provenance\n")
    if isinstance(data_cfg, Mapping):
        md_parts.append(f"- **source**: {data_cfg.get('source')}")
        md_parts.append(f"- **tickers**: {data_cfg.get('tickers')}")
        md_parts.append(f"- **start_date**: {data_cfg.get('start_date')}")
        md_parts.append(f"- **end_date**: {data_cfg.get('end_date')}")
        md_parts.append(f"- **output_csv**: `{data_cfg.get('output_csv')}`")
    md_parts.append("")
    md_parts.append("## Methodology (what was computed)\n")
    md_parts.append(
        "\n".join(
            [
                "- **Data**: Daily close prices downloaded from Stooq into a wide CSV, then converted into returns + rolling stats.",
                "- **Environment**: At each step t, the policy outputs target weights w_t.",
                "- **Turnover**: sum_i |w_{t,i} - w_{t-1,i}|.",
                "- **Costs**: `transaction_cost_rate * turnover` plus `slippage_rate * turnover` (applied to equity).",
                "- **Reward**: approximately `portfolio_return - cost - slippage - turnover_penalty` (per-step).",
                "- **Metrics**: computed over the episode and also within regimes (high_vol/low_vol/crash/calm).",
                "- **Walk-forward**: repeated out-of-sample folds with expanding training window; reported Sharpe distribution across folds.",
            ]
        )
    )
    md_parts.append("")
    md_parts.append("## Metrics glossary (how to read the tables)\n")
    md_parts.append(
        "\n".join(
            [
                "- **final_equity**: ending portfolio value after the episode. **Higher is better**.",
                "- **total_reward**: sum of per-step rewards. In this MVP env, reward is approximately: "
                "`portfolio_return - transaction_cost - slippage - turnover_penalty`. **Higher is better**.",
                "- **sharpe**: mean(returns)/std(returns) over steps. **Higher is better**. Can be NaN if std is ~0.",
                "- **max_drawdown**: worst peak→trough equity drop fraction (e.g. 0.25 = -25%). **Lower is better**.",
                "- **cvar_5**: average of the worst 5% returns (tail risk). **Less negative / higher is better**.",
                "- **avg_turnover**: average Σ|w_t − w_{t−1}|. High turnover implies high trading/costs. **Lower is better**.",
                "- **walk-forward (folds/mean/median/p10/p90 sharpe)**: out-of-sample Sharpe distribution across folds. "
                "Prefer higher median and a less-negative p10 (worst-case periods).",
                "- **regimes (high_vol/low_vol/crash/calm)**: metrics computed only on those slices. Use to see fragility.",
            ]
        )
    )
    md_parts.append("")
    md_parts.append("## Key takeaways\n")
    md_parts.append("\n".join(takeaways) if takeaways else "_(none)_")
    md_parts.append("## Benchmark comparison\n")
    md_parts.append(
        "_Interpretation_: compare `final_equity` and risk metrics (`max_drawdown`, `cvar_5`). "
        "A strategy can look great on equity but still be fragile in tail risk."
    )
    md_parts.append(markdown_table(comparison, columns=bench_cols))

    # Per-policy cost diagnostics from eval series
    bench_dir = Path(cfg.benchmarks_comparison_path).parent
    cost_rows = []
    if isinstance(comparison, list):
        for row in comparison:
            if not isinstance(row, Mapping):
                continue
            policy = row.get("policy")
            if not isinstance(policy, str):
                continue
            eval_path = bench_dir / f"{policy}.eval.json"
            if not eval_path.exists():
                continue
            try:
                obj = json.loads(eval_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            turnover = _safe_float_series(obj, "turnover")
            cost = _safe_float_series(obj, "cost")
            slippage = _safe_float_series(obj, "slippage")
            turn_pen = _safe_float_series(obj, "turnover_penalty")
            cost_rows.append(
                {
                    "policy": policy,
                    "steps": len(cost),
                    "avg_turnover": _mean(turnover),
                    "total_cost_rate": _sum(cost),
                    "total_slippage_rate": _sum(slippage),
                    "total_turnover_penalty": _sum(turn_pen),
                }
            )
    if cost_rows:
        md_parts.append("\n## Trading frictions diagnostics (from episode series)\n")
        md_parts.append(
            "_Interpretation_: these are **rate terms** per-step aggregated over the episode (not dollar costs). "
            "Higher turnover/costs indicate less realistic trading behavior._"
        )
        md_parts.append(
            markdown_table(
                cost_rows,
                columns=[
                    "policy",
                    "steps",
                    "avg_turnover",
                    "total_cost_rate",
                    "total_slippage_rate",
                    "total_turnover_penalty",
                ],
            )
        )

    # Red flags (expanded heuristics with sub-parts)
    thr = RedFlagThresholds(
        crash_sharpe_max=cfg.crash_sharpe_max,
        max_drawdown_max=cfg.max_drawdown_max,
        cvar5_min=cfg.cvar5_min,
        avg_turnover_max=cfg.avg_turnover_max,
        total_cost_rate_max=cfg.total_cost_rate_max,
        pct_pos_sharpe_min=cfg.pct_pos_sharpe_min,
        p10_sharpe_min=cfg.p10_sharpe_min,
    )
    red_regime: list[str] = []
    red_risk: list[str] = []
    red_cost: list[str] = []
    red_wf: list[str] = []

    # Helper maps from comparison/stress/cost/wf tables.
    comparison_by_policy = {
        str(r.get("policy")): r for r in comparison if isinstance(r, Mapping) and isinstance(r.get("policy"), str)
    } if isinstance(comparison, list) else {}

    # stress_rows contains one row per policy/regime.
    for r in stress_rows:
        pol = r.get("policy")
        reg = r.get("regime")
        if not isinstance(pol, str) or not isinstance(reg, str):
            continue
        if reg == "crash":
            s = r.get("sharpe")
            if isinstance(s, (int, float)) and float(s) < thr.crash_sharpe_max:
                red_regime.append(
                    f"- `{pol}` crash Sharpe is {float(s):.3g} (threshold: >= {thr.crash_sharpe_max})."
                )

    for pol, r in comparison_by_policy.items():
        mdd = r.get("max_drawdown")
        cv = r.get("cvar_5")
        if isinstance(mdd, (int, float)) and float(mdd) > thr.max_drawdown_max:
            red_risk.append(
                f"- `{pol}` max_drawdown is {float(mdd):.3g} (threshold: <= {thr.max_drawdown_max})."
            )
        if isinstance(cv, (int, float)) and float(cv) < thr.cvar5_min:
            red_risk.append(
                f"- `{pol}` cvar_5 is {float(cv):.3g} (threshold: >= {thr.cvar5_min})."
            )

    for r in cost_rows:
        pol = r.get("policy")
        if not isinstance(pol, str):
            continue
        at = r.get("avg_turnover")
        tc = r.get("total_cost_rate")
        if isinstance(at, (int, float)) and float(at) > thr.avg_turnover_max:
            red_cost.append(
                f"- `{pol}` avg_turnover is {float(at):.3g} (threshold: <= {thr.avg_turnover_max})."
            )
        if isinstance(tc, (int, float)) and float(tc) > thr.total_cost_rate_max:
            red_cost.append(
                f"- `{pol}` total_cost_rate is {float(tc):.3g} (threshold: <= {thr.total_cost_rate_max})."
            )

    for r in wf_rows:
        pol = r.get("policy")
        if not isinstance(pol, str):
            continue
        p10 = r.get("p10_sharpe")
        pct_pos = r.get("pct_pos_sharpe")
        if isinstance(p10, (int, float)) and float(p10) < thr.p10_sharpe_min:
            red_wf.append(
                f"- `{pol}` p10_sharpe is {float(p10):.3g} (threshold: >= {thr.p10_sharpe_min})."
            )
        if isinstance(pct_pos, (int, float)) and float(pct_pos) < thr.pct_pos_sharpe_min:
            red_wf.append(
                f"- `{pol}` pct_pos_sharpe is {float(pct_pos):.3g}% (threshold: >= {thr.pct_pos_sharpe_min}%)."
            )

    red_any = any([red_regime, red_risk, red_cost, red_wf])
    md_parts.append("\n## Red flags (automatic heuristics)\n")
    md_parts.append(
        "_Thresholds used_: "
        f"crash_sharpe>={thr.crash_sharpe_max}, max_drawdown<={thr.max_drawdown_max}, "
        f"cvar_5>={thr.cvar5_min}, avg_turnover<={thr.avg_turnover_max}, total_cost_rate<={thr.total_cost_rate_max}, "
        f"pct_pos_sharpe>={thr.pct_pos_sharpe_min}%, p10_sharpe>={thr.p10_sharpe_min}."
    )
    md_parts.append("\n### 1) Regime fragility (crash behavior)\n")
    md_parts.append("\n".join(red_regime) if red_regime else "_(none triggered)_")
    md_parts.append("\n### 2) Drawdown & tail-risk warnings\n")
    md_parts.append("\n".join(red_risk) if red_risk else "_(none triggered)_")
    md_parts.append("\n### 3) Turnover & trading-cost warnings\n")
    md_parts.append("\n".join(red_cost) if red_cost else "_(none triggered)_")
    md_parts.append("\n### 4) Walk-forward stability warnings\n")
    md_parts.append("\n".join(red_wf) if red_wf else "_(none triggered)_")
    md_parts.append("\n### 5) Summary\n")
    md_parts.append("- Red flags triggered." if red_any else "- No red flags triggered.")
    md_parts.append("\n## Walk-forward robustness (Sharpe across folds)\n")
    md_parts.append(
        "_Interpretation_: higher `median_sharpe` is good; very negative `p10_sharpe` indicates instability in some periods._"
    )
    md_parts.append(markdown_table(wf_rows, columns=wf_cols))

    # Walk-forward worst folds diagnostic
    worst_rows = []
    if isinstance(folds, list) and folds:
        for fold in folds:
            if not isinstance(fold, Mapping):
                continue
            fold_idx = fold.get("fold")
            results = fold.get("results")
            if not isinstance(results, list):
                continue
            for r in results:
                if not isinstance(r, Mapping):
                    continue
                policy = r.get("policy")
                summ = r.get("summary", {})
                if not isinstance(policy, str) or not isinstance(summ, Mapping):
                    continue
                s = summ.get("sharpe")
                fe = summ.get("final_equity")
                if isinstance(s, (int, float)):
                    split = r.get("split", {})
                    test_start = None
                    test_end = None
                    if isinstance(split, Mapping):
                        ts = split.get("test_start")
                        te = split.get("test_end")
                        test_start = int(ts) if isinstance(ts, int) else None
                        test_end = int(te) if isinstance(te, int) else None
                    start_date = feature_dates[test_start] if (test_start is not None and test_start < len(feature_dates)) else None
                    end_date = (
                        feature_dates[test_end - 1]
                        if (test_end is not None and test_end - 1 < len(feature_dates) and test_end - 1 >= 0)
                        else None
                    )
                    worst_rows.append(
                        {
                            "policy": policy,
                            "fold": fold_idx,
                            "sharpe": float(s),
                            "final_equity": float(fe) if isinstance(fe, (int, float)) else None,
                            "test_start": test_start,
                            "test_end": test_end,
                            "test_start_date": start_date,
                            "test_end_date": end_date,
                        }
                    )
        # take worst 5 per policy by sharpe
        per_pol: dict[str, list[dict[str, Any]]] = {}
        for r in worst_rows:
            per_pol.setdefault(r["policy"], []).append(r)
        flat = []
        for pol, rows in per_pol.items():
            rows_sorted = sorted(rows, key=lambda x: float(x["sharpe"]))
            flat.extend(rows_sorted[:5])
        if flat:
            md_parts.append("\n## Walk-forward worst folds (lowest Sharpe)\n")
            md_parts.append(
                "_Interpretation_: these are the periods where a strategy performed worst out-of-sample. "
                "If these are extremely negative, the strategy is unstable._"
            )
            md_parts.append(
                markdown_table(
                    flat,
                    columns=[
                        "policy",
                        "fold",
                        "sharpe",
                        "final_equity",
                        "test_start_date",
                        "test_end_date",
                    ],
                )
            )
    md_parts.append("\n## Regime / stress breakdown\n")
    md_parts.append(
        "_Interpretation_: focus on `crash` and `high_vol`. If performance collapses there, the strategy is not robust._"
    )
    md_parts.append(markdown_table(stress_rows, columns=stress_cols))

    if rl_eval is not None:
        md_parts.append("\n## RL policy (SB3) evaluation\n")
        md_parts.append("Summary:\n")
        md_parts.append("```json\n" + json.dumps(rl_eval.get("summary", {}), indent=2) + "\n```")

    md_path = out_dir / "report.md"
    md_path.write_text("\n".join(md_parts), encoding="utf-8")

    # HTML report
    body = []
    body.append("<h1>RL-PortfolioLab — Baseline Report</h1>")
    body.append('<p class="muted">Generated from Phase 4 benchmarks + Phase 5 stress slicing + Phase 7 walk-forward.</p>')
    body.append(
        "<p class=\"muted\">Jump to: "
        "<a href=\"#bench\">benchmarks</a> · "
        "<a href=\"#wf\">walk-forward</a> · "
        "<a href=\"#stress\">stress</a>"
        "</p>"
    )

    if isinstance(data_cfg, Mapping):
        body.append("<h2>Dataset & provenance</h2>")
        body.append(
            "<pre>"
            + json.dumps(
                {
                    "source": data_cfg.get("source"),
                    "tickers": data_cfg.get("tickers"),
                    "start_date": data_cfg.get("start_date"),
                    "end_date": data_cfg.get("end_date"),
                    "output_csv": data_cfg.get("output_csv"),
                },
                indent=2,
            )
            + "</pre>"
        )

    body.append("<h2>Metrics glossary (how to read the tables)</h2>")
    body.append(
        "<ul>"
        + "".join(
            [
                "<li><b>final_equity</b>: ending portfolio value. <b>Higher is better</b>.</li>",
                "<li><b>total_reward</b>: sum of per-step rewards ≈ return − costs − turnover penalty. <b>Higher is better</b>.</li>",
                "<li><b>sharpe</b>: mean(returns)/std(returns). <b>Higher is better</b> (NaN if std≈0).</li>",
                "<li><b>max_drawdown</b>: worst peak→trough equity drop fraction. <b>Lower is better</b>.</li>",
                "<li><b>cvar_5</b>: average of worst 5% returns. <b>Less negative / higher is better</b>.</li>",
                "<li><b>avg_turnover</b>: avg Σ|Δw| per step (trading aggressiveness). <b>Lower is better</b>.</li>",
                "<li><b>walk-forward sharpe</b>: distribution across out-of-sample folds; prefer higher median and better p10.</li>",
                "<li><b>regimes</b>: metrics computed within high-vol/low-vol/crash/calm slices.</li>",
            ]
        )
        + "</ul>"
    )

    if takeaways:
        body.append("<h2>Key takeaways</h2>")
        body.append("<ul>" + "".join(f"<li>{t[2:]}</li>" for t in takeaways) + "</ul>")

    body.append("<h2>Methodology</h2>")
    body.append(
        "<ul>"
        + "".join(
            [
                "<li><b>Data</b>: daily close prices → returns + rolling stats.</li>",
                "<li><b>Action</b>: continuous target weights per asset.</li>",
                "<li><b>Turnover</b>: Σ|w_t − w_{t−1}|.</li>",
                "<li><b>Costs</b>: transaction_cost_rate * turnover + slippage_rate * turnover (applied to equity).</li>",
                "<li><b>Reward</b>: return − cost − slippage − turnover_penalty (per-step).</li>",
                "<li><b>Walk-forward</b>: expanding train window; evaluate Sharpe distribution across OOS folds.</li>",
            ]
        )
        + "</ul>"
    )
    body.append("<h2 id=\"bench\">Benchmark comparison</h2>")
    body.append(
        '<p class="muted">Compare final equity and risk metrics. A strategy can "win" on equity but still be fragile (drawdown/CVaR).</p>'
    )
    body.append(html_table(comparison, columns=bench_cols))

    # Equity curve cards (sparklines) by loading per-policy eval files if they exist.
    bench_dir = Path(cfg.benchmarks_comparison_path).parent
    cards = []
    if isinstance(comparison, list):
        for row in comparison:
            if not isinstance(row, Mapping):
                continue
            policy = row.get("policy")
            if not isinstance(policy, str):
                continue
            eval_path = bench_dir / f"{policy}.eval.json"
            equity = _safe_equity_series_from_eval(eval_path) if eval_path.exists() else []
            if equity:
                cards.append(
                    "<div class=\"card\">"
                    f"<h3>{policy}</h3>"
                    f"{sparkline_svg(equity)}"
                    f"<div class=\"muted\">final_equity: {row.get('final_equity')}</div>"
                    "</div>"
                )
    if cards:
        body.append("<h2>Equity curves (benchmarks)</h2>")
        body.append('<p class="muted">Sparklines show the equity path over the episode (downsampled).</p>')
        body.append("<div class=\"grid\">" + "".join(cards) + "</div>")

    if cost_rows:
        body.append("<h2>Trading frictions diagnostics</h2>")
        body.append(
            '<p class="muted">These are aggregated per-step rate terms (not dollars). High turnover usually implies higher costs and lower realism.</p>'
        )
        body.append(
            html_table(
                cost_rows,
                columns=[
                    "policy",
                    "steps",
                    "avg_turnover",
                    "total_cost_rate",
                    "total_slippage_rate",
                    "total_turnover_penalty",
                ],
            )
        )

    body.append("<h2 id=\"wf\">Walk-forward robustness (Sharpe across folds)</h2>")
    body.append('<p class="muted">Higher median Sharpe is better. Watch p10 (bad periods) and p90 (good periods).</p>')
    body.append(html_table(wf_rows, columns=wf_cols))

    if isinstance(folds, list) and folds:
        # show worst 5 folds per policy (same as markdown)
        if 'flat' in locals() and flat:
            body.append("<h2>Walk-forward worst folds (lowest Sharpe)</h2>")
            body.append('<p class="muted">Where each policy performed worst out-of-sample.</p>')
            cols = ["policy", "fold", "sharpe", "final_equity"]
            if feature_dates:
                cols += ["test_start_date", "test_end_date"]
            body.append(html_table(flat, columns=cols))

    body.append("<h2>Red flags (automatic heuristics)</h2>")
    body.append(
        "<p class=\"muted\">Thresholds used: "
        f"crash_sharpe&gt;={thr.crash_sharpe_max}, max_drawdown&lt;={thr.max_drawdown_max}, "
        f"cvar_5&gt;={thr.cvar5_min}, avg_turnover&lt;={thr.avg_turnover_max}, total_cost_rate&lt;={thr.total_cost_rate_max}, "
        f"pct_pos_sharpe&gt;={thr.pct_pos_sharpe_min}%, p10_sharpe&gt;={thr.p10_sharpe_min}."
        "</p>"
    )
    body.append("<h3>1) Regime fragility (crash behavior)</h3>")
    body.append(
        "<ul>" + "".join(f"<li>{x[2:]}</li>" for x in red_regime) + "</ul>"
        if red_regime
        else '<p class="muted">(none triggered)</p>'
    )
    body.append("<h3>2) Drawdown & tail-risk warnings</h3>")
    body.append(
        "<ul>" + "".join(f"<li>{x[2:]}</li>" for x in red_risk) + "</ul>"
        if red_risk
        else '<p class="muted">(none triggered)</p>'
    )
    body.append("<h3>3) Turnover & trading-cost warnings</h3>")
    body.append(
        "<ul>" + "".join(f"<li>{x[2:]}</li>" for x in red_cost) + "</ul>"
        if red_cost
        else '<p class="muted">(none triggered)</p>'
    )
    body.append("<h3>4) Walk-forward stability warnings</h3>")
    body.append(
        "<ul>" + "".join(f"<li>{x[2:]}</li>" for x in red_wf) + "</ul>"
        if red_wf
        else '<p class="muted">(none triggered)</p>'
    )
    body.append("<h3>5) Summary</h3>")
    body.append("<p>" + ("Red flags triggered." if red_any else "No red flags triggered.") + "</p>")
    body.append("<h2 id=\"stress\">Regime / stress breakdown</h2>")
    body.append('<p class="muted">Focus on crash/high-vol regimes to assess robustness.</p>')
    body.append(html_table(stress_rows, columns=stress_cols))
    if rl_eval is not None:
        body.append("<h2>RL policy (SB3) evaluation</h2>")
        body.append("<pre>" + json.dumps(rl_eval.get("summary", {}), indent=2) + "</pre>")

    html_path = out_dir / "report.html"

    # RL equity curve card (optional)
    if cfg.rl_eval_path is not None:
        rl_path = Path(cfg.rl_eval_path)
        if rl_path.exists():
            rl_equity = _safe_equity_series_from_eval(rl_path)
            if rl_equity:
                body.append("<h2>Equity curve (RL eval)</h2>")
                body.append("<div class=\"grid\"><div class=\"card\"><h3>RL (SB3)</h3>" + sparkline_svg(rl_equity, stroke="#d62728") + "</div></div>")

    html_path.write_text(html_page(title="RL-PortfolioLab Report", body_html="\n".join(body)), encoding="utf-8")

    return md_path, html_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 6 report (Markdown + HTML).")
    parser.add_argument("--config", required=True, help="Path to configs/phase6_report.yaml")
    args = parser.parse_args()

    cfg = load_phase6_config(args.config)
    md_path, html_path = run_phase6(cfg)
    print(f"Wrote report: {md_path}")
    print(f"Wrote report: {html_path}")


if __name__ == "__main__":
    main()

