from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from urllib.request import urlopen

import yaml


@dataclass(frozen=True)
class FetchConfig:
    tickers: list[str]
    start_date: str
    end_date: Optional[str]
    price_field: str
    output_csv: str
    date_column: str


def _require_mapping(x: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(x, Mapping):
        raise ValueError(f"`{name}` must be an object/mapping.")
    return x


def _require_str(x: Any, name: str) -> str:
    if not isinstance(x, str):
        raise ValueError(f"`{name}` must be a string.")
    return x


def load_fetch_config(path: str | Path) -> FetchConfig:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    root = _require_mapping(raw, "root")

    tickers = root.get("tickers")
    if not isinstance(tickers, list) or not all(isinstance(t, str) for t in tickers) or not tickers:
        raise ValueError("`tickers` must be a non-empty list of strings.")

    start_date = _require_str(root.get("start_date"), "start_date")
    end_date_raw = root.get("end_date")
    end_date = end_date_raw if isinstance(end_date_raw, str) else None

    return FetchConfig(
        tickers=list(tickers),
        start_date=start_date,
        end_date=end_date,
        price_field=_require_str(root.get("price_field"), "price_field"),
        output_csv=_require_str(root.get("output_csv"), "output_csv"),
        date_column=_require_str(root.get("date_column"), "date_column"),
    )


def _download_stooq_csv(ticker: str) -> list[dict[str, str]]:
    # Stooq daily historical CSV endpoint
    url = f"https://stooq.com/q/d/l/?s={ticker}&i=d"
    with urlopen(url) as resp:
        text = resp.read().decode("utf-8")
    reader = csv.DictReader(text.splitlines())
    rows = []
    for r in reader:
        if not r:
            continue
        rows.append({k.lower(): (v or "") for k, v in r.items()})
    return rows


def _in_date_range(d: str, start: str, end: Optional[str]) -> bool:
    if d < start:
        return False
    if end is not None and d > end:
        return False
    return True


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] != "--config":
        print("Usage: python scripts/fetch_prices_wide.py --config configs/data_stooq.yaml")
        sys.exit(2)

    cfg = load_fetch_config(sys.argv[2])

    all_dates: set[str] = set()
    series: dict[str, dict[str, float]] = {}

    for t in cfg.tickers:
        rows = _download_stooq_csv(t)
        per_date: dict[str, float] = {}
        for r in rows:
            date = r.get("date", "")
            if not date:
                continue
            if not _in_date_range(date, cfg.start_date, cfg.end_date):
                continue
            val = r.get(cfg.price_field.lower(), "")
            if val in ("", "nan", "NaN", "null"):
                continue
            try:
                per_date[date] = float(val)
                all_dates.add(date)
            except ValueError:
                continue
        if not per_date:
            raise RuntimeError(f"No data downloaded for ticker {t}. Check ticker symbol.")
        series[t] = per_date

    dates_sorted = sorted(all_dates)

    out_path = Path(cfg.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Wide CSV output: date + one column per ticker.
    with out_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [cfg.date_column] + cfg.tickers
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in dates_sorted:
            row: dict[str, str] = {cfg.date_column: d}
            for t in cfg.tickers:
                v = series[t].get(d)
                row[t] = "" if v is None else str(v)
            writer.writerow(row)

    print(f"Wrote {out_path} with {len(dates_sorted)} rows and {len(cfg.tickers)} tickers.")


if __name__ == "__main__":
    main()

