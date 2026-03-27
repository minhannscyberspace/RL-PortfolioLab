from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml


@dataclass(frozen=True)
class FeatureBuilderConfig:
    return_lag: int
    vol_window: int
    cov_window: int
    min_periods: int


@dataclass(frozen=True)
class Phase1WideConfig:
    seed: int
    date_column: str
    asset_columns: Optional[list[str]]
    missing_value_strategy: str
    feature_builder: FeatureBuilderConfig


def _require_int(value: Any, name: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"Config field `{name}` must be an int; got {type(value).__name__}")
    return value


def _require_str(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Config field `{name}` must be a str; got {type(value).__name__}")
    return value


def load_yaml_config(path: str | Path) -> Phase1WideConfig:
    """Load and validate Phase 1 wide-CSV config."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, Mapping):
        raise ValueError("Config root must be a mapping/object.")

    seed = _require_int(raw.get("seed"), "seed")
    date_column = _require_str(raw.get("date_column"), "date_column")
    asset_columns_raw = raw.get("asset_columns")
    missing_value_strategy = _require_str(raw.get("missing_value_strategy"), "missing_value_strategy")

    asset_columns: Optional[list[str]]
    if asset_columns_raw is None:
        asset_columns = None
    elif isinstance(asset_columns_raw, list) and all(isinstance(x, str) for x in asset_columns_raw):
        asset_columns = list(asset_columns_raw)
    else:
        raise ValueError("Config field `asset_columns` must be null or a list of strings.")

    fb_raw = raw.get("feature_builder")
    if not isinstance(fb_raw, Mapping):
        raise ValueError("Config field `feature_builder` must be an object.")

    fb = FeatureBuilderConfig(
        return_lag=_require_int(fb_raw.get("return_lag"), "feature_builder.return_lag"),
        vol_window=_require_int(fb_raw.get("vol_window"), "feature_builder.vol_window"),
        cov_window=_require_int(fb_raw.get("cov_window"), "feature_builder.cov_window"),
        min_periods=_require_int(fb_raw.get("min_periods"), "feature_builder.min_periods"),
    )

    if fb.return_lag < 1:
        raise ValueError("feature_builder.return_lag must be >= 1")
    if fb.vol_window < 2:
        raise ValueError("feature_builder.vol_window must be >= 2")
    if fb.cov_window < 2:
        raise ValueError("feature_builder.cov_window must be >= 2")
    if fb.min_periods < 1:
        raise ValueError("feature_builder.min_periods must be >= 1")

    return Phase1WideConfig(
        seed=seed,
        date_column=date_column,
        asset_columns=asset_columns,
        missing_value_strategy=missing_value_strategy,
        feature_builder=fb,
    )

