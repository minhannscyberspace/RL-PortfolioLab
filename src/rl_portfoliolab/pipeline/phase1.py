from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from dataclasses import asdict

from rl_portfoliolab.data.alignment import align_price_matrix
from rl_portfoliolab.data.loader import load_market_data_wide
from rl_portfoliolab.data.types import WideMarketData
from rl_portfoliolab.features.builder import build_features_from_prices
from rl_portfoliolab.utils.config import load_yaml_config, Phase1WideConfig
from rl_portfoliolab.utils.seeding import set_seed


def _generate_synthetic_wide_market_data(*, seed: int, n_days: int, assets: list[str]) -> WideMarketData:
    """
    Deterministic synthetic prices using only Python stdlib (no NumPy).

    Uses a random walk in log-space to keep prices positive.
    """
    set_seed(seed)
    # Generate daily dates starting from 2020-01-01.
    from datetime import datetime, timedelta

    start = datetime(2020, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n_days)]

    # Random walk in log space.
    log_price = [math.log(random.uniform(50, 150)) for _ in assets]
    prices: list[list[float]] = []
    for _ in range(n_days):
        row: list[float] = []
        for j in range(len(assets)):
            log_price[j] += random.gauss(0.0, 0.01)
            row.append(math.exp(log_price[j]))
        prices.append(row)

    return WideMarketData(dates=dates, assets=list(assets), prices=prices)


def _nan_to_none(x: float) -> float | None:
    return None if isinstance(x, float) and math.isnan(x) else x


def _serialize_features(features: object) -> dict:
    # FeatureArrays uses nested lists; convert NaN floats -> None for JSON safety.
    from rl_portfoliolab.features.builder import FeatureArrays  # local import

    if not isinstance(features, FeatureArrays):
        raise TypeError("Expected FeatureArrays")

    dates = [d.isoformat() for d in features.dates]  # type: ignore[attr-defined]
    returns = [[_nan_to_none(v) for v in row] for row in features.returns]
    volatility = [[_nan_to_none(v) for v in row] for row in features.volatility]
    covariance = [
        [[_nan_to_none(v) for v in row_j] for row_j in cov_i]
        for cov_i in features.covariance
    ]

    return {
        "dates": dates,
        "assets": features.assets,
        "returns": returns,
        "volatility": volatility,
        "covariance": covariance,
    }


def run_phase1_wide(
    *,
    csv_path: str | Path | None,
    config_path: str | Path,
    output_dir: str | Path,
) -> Path:
    config: Phase1WideConfig = load_yaml_config(config_path)
    set_seed(config.seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # For now, if csv_path is not provided, generate deterministic synthetic data.
    if csv_path is None:
        assets = ["A0", "A1", "A2", "A3"]
        market_data = _generate_synthetic_wide_market_data(seed=config.seed, n_days=120, assets=assets)
    else:
        market_data = load_market_data_wide(
            csv_path,
            date_column=config.date_column,
            asset_columns=config.asset_columns,
        )

    aligned = align_price_matrix(market_data, missing_value_strategy=config.missing_value_strategy)
    features = build_features_from_prices(
        aligned,
        return_lag=config.feature_builder.return_lag,
        vol_window=config.feature_builder.vol_window,
        cov_window=config.feature_builder.cov_window,
        min_periods=config.feature_builder.min_periods,
    )

    # Save artifacts with explicit provenance.
    out_file = output_path / "phase1_wide_features.json"
    payload = {
        "config": dict(asdict(config)),
        **_serialize_features(features),
    }
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Also save a small summary for human verification.
    summary_path = output_path / "phase1_wide_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Phase 1 Summary (wide)\n")
        f.write(f"Config: {config}\n")
        f.write(f"Aligned price shape (T_all, N): ({len(aligned.dates)}, {len(aligned.assets)})\n")
        f.write(f"Features returns shape (T, N): ({len(features.returns)}, {len(features.assets)})\n")
        f.write(
            f"Features volatility shape (T, N): ({len(features.volatility)}, {len(features.assets)})\n"
        )
        f.write(
            f"Features covariance shape (T, N, N): ({len(features.covariance)}, {len(features.assets)}, {len(features.assets)})\n"
        )
        f.write("Done.\n")

    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 (wide format) pipeline.")
    parser.add_argument("--config", required=True, help="Path to configs/phase1_wide.yaml")
    parser.add_argument("--csv", required=False, default=None, help="Wide CSV path (optional)")
    parser.add_argument("--out", required=False, default="artifacts/phase1", help="Output directory")
    args = parser.parse_args()

    run_phase1_wide(csv_path=args.csv, config_path=args.config, output_dir=args.out)


if __name__ == "__main__":
    main()

