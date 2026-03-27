from __future__ import annotations

import csv
from datetime import datetime

from rl_portfoliolab.data.loader import load_market_data_wide


def test_load_market_data_wide_infers_assets(tmp_path):
    csv_path = tmp_path / "prices.csv"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "A0", "A1"])
        writer.writeheader()
        # Unsorted rows (wide format loader should sort by date).
        writer.writerow({"date": "2020-01-02", "A0": "101.0", "A1": "201.0"})
        writer.writerow({"date": "2020-01-01", "A0": "100.0", "A1": "200.0"})
        writer.writerow({"date": "2020-01-03", "A0": "102.0", "A1": "202.0"})

    out = load_market_data_wide(csv_path, date_column="date", asset_columns=None)

    assert out.assets == ["A0", "A1"]
    assert out.dates[0] == datetime.fromisoformat("2020-01-01")
    assert out.dates[1] == datetime.fromisoformat("2020-01-02")
    assert out.dates[2] == datetime.fromisoformat("2020-01-03")
    assert len(out.prices) == 3
    assert len(out.prices[0]) == 2
    assert out.prices[0][0] == 100.0  # A0 at 2020-01-01


def test_load_market_data_wide_rejects_missing_assets(tmp_path):
    csv_path = tmp_path / "prices.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "A0"])
        writer.writeheader()
        writer.writerow({"date": "2020-01-01", "A0": "100.0"})
        writer.writerow({"date": "2020-01-02", "A0": "101.0"})

    try:
        load_market_data_wide(csv_path, date_column="date", asset_columns=["A0", "A1"])
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "asset_columns missing from CSV" in str(e)

