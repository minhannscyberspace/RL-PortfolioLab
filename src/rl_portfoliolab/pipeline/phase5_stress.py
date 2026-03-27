from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from rl_portfoliolab.evaluation.metrics import cvar, max_drawdown, sharpe, turnover_from_weights
from rl_portfoliolab.stress.regimes import apply_mask, classify_regimes
from rl_portfoliolab.utils.seeding import set_seed


@dataclass(frozen=True)
class Phase5Config:
    seed: int
    input_eval_dir: str
    vol_window: int
    high_vol_quantile: float
    crash_quantile: float
    out_dir: str
    run_name: str


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


def _require_float(x: Any, name: str) -> float:
    if not isinstance(x, (int, float)):
        raise ValueError(f"`{name}` must be a number.")
    return float(x)


def load_phase5_config(path: str | Path) -> Phase5Config:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    root = _require_mapping(raw, "root")

    regimes = _require_mapping(root.get("regimes"), "regimes")
    output = _require_mapping(root.get("output"), "output")

    return Phase5Config(
        seed=_require_int(root.get("seed"), "seed"),
        input_eval_dir=_require_str(root.get("input_eval_dir"), "input_eval_dir"),
        vol_window=_require_int(regimes.get("vol_window"), "regimes.vol_window"),
        high_vol_quantile=_require_float(regimes.get("high_vol_quantile"), "regimes.high_vol_quantile"),
        crash_quantile=_require_float(regimes.get("crash_quantile"), "regimes.crash_quantile"),
        out_dir=_require_str(output.get("out_dir"), "output.out_dir"),
        run_name=_require_str(output.get("run_name"), "output.run_name"),
    )


def _load_eval_file(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "series" not in raw or not isinstance(raw["series"], list):
        raise ValueError(f"Eval file missing `series`: {path}")
    return raw


def _extract_series(eval_obj: dict[str, Any]) -> tuple[list[Optional[float]], list[Optional[float]], list[list[Optional[float]]]]:
    # returns: portfolio_return per step, equity_after per step, weights per step (optional)
    rets: list[Optional[float]] = []
    equity: list[Optional[float]] = []
    weights: list[list[Optional[float]]] = []

    for row in eval_obj["series"]:
        if not isinstance(row, Mapping):
            continue
        rets.append(row.get("portfolio_return"))
        equity.append(row.get("equity_after"))
        w = row.get("weights")
        # Many eval series don't store weights; allow missing.
        if isinstance(w, list):
            weights.append([float(x) if x is not None else None for x in w])

    return rets, equity, weights


def _summarize(rets: list[Optional[float]], equity: list[Optional[float]], weights: list[list[Optional[float]]]) -> dict[str, Any]:
    return {
        "steps": len([x for x in rets if x is not None]),
        "sharpe": sharpe(rets),
        "max_drawdown": max_drawdown(equity),
        "cvar_5": cvar(rets, alpha=0.05),
        "avg_turnover": turnover_from_weights(weights) if weights else None,
        "final_equity": (float(equity[-1]) if equity and equity[-1] is not None else None),
    }


def run_phase5(cfg: Phase5Config) -> Path:
    set_seed(cfg.seed)
    in_dir = Path(cfg.input_eval_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f"input_eval_dir not found: {in_dir}")

    out_dir = Path(cfg.out_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_files = sorted([p for p in in_dir.glob("*.eval.json") if p.is_file()])
    if not eval_files:
        raise ValueError(f"No *.eval.json files found in {in_dir}")

    all_results = []
    for p in eval_files:
        obj = _load_eval_file(p)
        rets, equity, weights = _extract_series(obj)
        masks = classify_regimes(
            rets,
            vol_window=cfg.vol_window,
            high_vol_quantile=cfg.high_vol_quantile,
            crash_quantile=cfg.crash_quantile,
        )

        out = {
            "file": str(p),
            "policy": obj.get("policy"),
            "overall": _summarize(rets, equity, weights),
            "regimes": {
                "high_vol": _summarize(apply_mask(rets, masks.high_vol), apply_mask(equity, masks.high_vol), []),
                "low_vol": _summarize(apply_mask(rets, masks.low_vol), apply_mask(equity, masks.low_vol), []),
                "crash": _summarize(apply_mask(rets, masks.crash), apply_mask(equity, masks.crash), []),
                "calm": _summarize(apply_mask(rets, masks.calm), apply_mask(equity, masks.calm), []),
            },
        }
        all_results.append(out)
        (out_dir / f"{p.stem}.stress.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    summary_path = out_dir / "stress_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 5 stress/regime slicing.")
    parser.add_argument("--config", required=True, help="Path to configs/phase5_stress.yaml")
    args = parser.parse_args()

    cfg = load_phase5_config(args.config)
    out = run_phase5(cfg)
    print(f"Wrote stress summary: {out}")


if __name__ == "__main__":
    main()

