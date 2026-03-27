# RL PortfolioLab 

## About

**RL PortfolioLab** is a Python research pipeline for reinforcement learning–based portfolio allocation. It connects:

- **Market data** (wide-format daily prices)
- **Feature engineering** (returns, rolling volatility, rolling covariance) with **no lookahead** in the feature builder tests
- A **custom portfolio environment** (long-only weights, gross exposure cap, transaction costs, slippage, turnover penalty)
- **RL training and evaluation** via **Stable-Baselines3** (**PPO** or **SAC**) through a **Gymnasium** adapter
- **Benchmark strategies** on the **same** environment: equal weight, momentum, inverse volatility, mean-variance (ridge-regularized)
- **Regime stress tests** (slice performance by volatility / crash-style regimes)
- **Walk-forward out-of-sample evaluation** for **benchmark policies** (expanding train window, fixed test window, stepped forward)
- **Automated reporting** (Markdown + HTML): comparisons, stress tables, walk-forward stats, optional RL eval summary, configurable **red-flag** thresholds

My goal is a credible, reproducible research loop: same configs drive the same artifact paths, and downstream phases read prior JSON outputs.

---

## Repository map

| Path | Role |
|------|------|
| `configs/` | YAML for every phase (data, features, env, train, eval, benchmarks, stress, walk-forward, report) |
| `scripts/` | Runnable entrypoints (`run_all_real_data.py`, `fetch_prices_wide.py`, `run_phase*.py`) |
| `src/rl_portfoliolab/` | Library code: data, features, envs, agents, benchmarks, evaluation, stress, reporting, pipelines |
| `data/raw/` | Fetched wide CSV (default: `prices_wide.csv` from Stooq config) |
| `artifacts/` | Generated JSON, models, summaries (by phase / run name) |
| `reports/` | Generated `report.md` and `report.html` (by report run name) |
| `tests/` | `pytest` suite for loader, features, env, metrics, regimes, walk-forward splits, report smoke tests, etc. |

---

## Prerequisites

Use a **virtual environment** and install dependencies needed for your workflow.

**Minimal (phases that avoid heavy ML stacks):** PyYAML is required for configs; `pytest` for tests. See `requirements.txt` for a full list.

**Full pipeline including RL (Phase 3):** you need a working **NumPy** + **PyTorch** stack and:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install pyyaml pytest gymnasium stable-baselines3 torch numpy
```

---

## Rerun everything in one command

From the repository root:

```bash
python scripts/run_all_real_data.py
```

This runs **in order**:

1. Fetch wide CSV — `scripts/fetch_prices_wide.py` → `configs/data_stooq.yaml`
2. Phase 1 — `scripts/run_phase1.py` → features JSON under `artifacts/phase1/`
3. Phase 2 — `scripts/run_phase2_env.py` — environment smoke episode
4. Phase 3 train — `scripts/run_phase3_train.py` — SB3 PPO/SAC
5. Phase 3 eval — `scripts/run_phase3_eval.py` — rollout + metrics JSON
6. Phase 4 — `scripts/run_phase4_benchmarks.py` — four benchmarks + `comparison.json`
7. Phase 5 — `scripts/run_phase5_stress.py` — regime stress summaries
8. Phase 7 — `scripts/run_phase7_walkforward.py` — OOS walk-forward for benchmarks
9. Phase 6 — `scripts/run_phase6_report.py` — final report

**Default outputs** (matching current configs):

- HTML report: `reports/baseline_report/report.html`
- Markdown report: `reports/baseline_report/report.md`

**Override config paths** (same defaults as above if omitted):

```bash
python scripts/run_all_real_data.py --help
```

Example with explicit configs:

```bash
python scripts/run_all_real_data.py \
  --data-config configs/data_stooq.yaml \
  --phase1-config configs/phase1_wide.yaml \
  --phase2-config configs/phase2_env.yaml \
  --phase3-train-config configs/phase3_train.yaml \
  --phase3-eval-config configs/phase3_eval.yaml \
  --phase4-config configs/phase4_benchmarks.yaml \
  --phase5-config configs/phase5_stress.yaml \
  --phase7-config configs/phase7_walkforward.yaml \
  --phase6-config configs/phase6_report.yaml \
  --phase1-csv data/raw/prices_wide.csv \
  --phase1-out artifacts/phase1
```

---

## Rerun step by step (exact commands)

Use this when debugging or regenerating only part of the pipeline.

```bash
# 0) Data: download wide CSV (Stooq)
python scripts/fetch_prices_wide.py --config configs/data_stooq.yaml

# 1) Features + Phase 1 artifacts
python scripts/run_phase1.py --config configs/phase1_wide.yaml \
  --csv data/raw/prices_wide.csv --out artifacts/phase1

# 2) Environment smoke test
python scripts/run_phase2_env.py --config configs/phase2_env.yaml

# 3) RL train + eval (requires numpy/torch/SB3)
python scripts/run_phase3_train.py --config configs/phase3_train.yaml
python scripts/run_phase3_eval.py --config configs/phase3_eval.yaml

# 4) Benchmarks
python scripts/run_phase4_benchmarks.py --config configs/phase4_benchmarks.yaml

# 5) Stress / regimes
python scripts/run_phase5_stress.py --config configs/phase5_stress.yaml

# 7) Walk-forward OOS (benchmarks)
python scripts/run_phase7_walkforward.py --config configs/phase7_walkforward.yaml

# 6) Report (reads phase 4, 5, 7 + optional RL eval)
python scripts/run_phase6_report.py --config configs/phase6_report.yaml
```

**Dependency note:** Phase 6 expects files listed in `configs/phase6_report.yaml` under `inputs:` (comparison JSON, stress summary, walk-forward summary, optional RL eval). Run phases **4 → 5 → 7** before **6**, or update paths if you use different `run_name` output folders.

---

## How to update behavior (what to edit, then what to rerun)

### Change tickers, dates, or output CSV

- **Edit:** `configs/data_stooq.yaml` (`tickers`, `start_date`, `end_date`, `price_field`, `output_csv`).
- **Rerun:** full pipeline from fetch, or at minimum fetch → phase1 → everything downstream.

### Change feature windows / missing data handling

- **Edit:** `configs/phase1_wide.yaml` (`missing_value_strategy`, `feature_builder.*`).
- **Rerun:** phase1 onward (or full `run_all_real_data.py`).

### Change portfolio constraints and trading frictions

- **Edit:** `configs/phase2_env.yaml` (`env.min_weight`, `env.max_weight`, `env.max_gross_exposure`, costs, slippage, turnover penalty). Ensure `phase1_features_path` points at your Phase 1 JSON.
- **Rerun:** phase2 onward (anything that uses the env or those features).

### Change RL algorithm, timesteps, or model output location

- **Edit:** `configs/phase3_train.yaml` (`algo.name` = `PPO` or `SAC`, `algo.total_timesteps`, `output.model_dir`, `output.run_name`).
- **Edit:** `configs/phase3_eval.yaml` so `model.model_path` matches the saved zip (default pattern: `artifacts/phase3/models/<run_name>/<algo>_model.zip`).
- **Rerun:** phase3 train → phase3 eval → phase6 (if you want RL section updated). Phase 5 can use phase3 eval if you point stress config at eval artifacts (see phase5 config comments).

### Change benchmarks

- **Edit:** `configs/phase4_benchmarks.yaml` (list under `benchmarks`, `lookback`, `ridge` for mean-variance).
- **Rerun:** phase4 → phase5 (if stress reads phase4 outputs) → phase7 → phase6.

### Change stress / regime thresholds

- **Edit:** `configs/phase5_stress.yaml` (`input_eval_dir`, `regimes.*`, `output.*`).
- **Rerun:** phase5 → phase6.

### Change walk-forward splits

- **Edit:** `configs/phase7_walkforward.yaml` (`splits.train_size`, `test_size`, `step_size`, benchmarks list, `output.*`).
- **Rerun:** phase7 → phase6.

### Change report inputs, red-flag thresholds, or report output folder

- **Edit:** `configs/phase6_report.yaml`:
  - `inputs.*_path` — must match where phase 4/5/7 (and optional phase3 eval) actually wrote files.
  - `red_flags.*` — numeric thresholds for automatic warnings.
  - `output.out_dir`, `output.run_name` — control `reports/<run_name>/`.
- **Rerun:** phase6 only (after upstream artifacts exist).

---

## Tests

From the repo root:

```bash
pytest -q
```

Core logic (loader, features, env reward/costs/constraints, metrics, regimes, walk-forward splits, report generation smoke tests) is covered without requiring a full SB3 training run.

---

## Design notes for maintainers

- **Benchmark walk-forward (Phase 7)** evaluates policies on **held-out time windows** defined in config. **RL training/eval (Phase 3)** uses the environment built from the **full** Phase 1 feature series unless you add a separate split—document this when publishing results.
- **Artifacts and reports** are intentional outputs; regenerate when configs change so provenance stays aligned (`AGENTS.md` marks these areas as sensitive for reproducibility).

---

## Quick reference: default artifact locations

| Stage | Typical outputs |
|-------|-----------------|
| Fetch | `data/raw/prices_wide.csv` (from `data_stooq.yaml`) |
| Phase 1 | `artifacts/phase1/` (e.g. `phase1_wide_features.json`) |
| Phase 3 train | `artifacts/phase3/models/baseline/ppo_model.zip` (names vary by algo/config) |
| Phase 3 eval | `artifacts/phase3/eval/baseline_eval/eval.json` |
| Phase 4 | `artifacts/phase4/benchmarks/baseline_benchmarks/` + `comparison.json` |
| Phase 5 | `artifacts/phase5/stress/baseline_stress/stress_summary.json` |
| Phase 7 | `artifacts/phase7/walkforward/baseline_walkforward/walkforward_summary.json` |
| Phase 6 | `reports/baseline_report/report.html`, `report.md` |

If you change `run_name` or `out_dir` in any YAML, update downstream configs (especially `phase6_report.yaml` `inputs`).

---

## Future work / extensions

Possible directions to grow RL-PortfolioLab beyond the current MVP:

- **Weight-centric API** — Single contract from features to execution: selection → allocation → timing → risk overlay, swappable modules (benchmarks, RL, rules) without rewriting the pipeline.
- **RL out-of-sample parity** — Walk-forward or rolling train/validate/test for SB3 policies (aligned with benchmark OOS splits), plus stricter split rules if feature windows overlap.
- **Data layer** — SQLite or Parquet cache, multiple providers behind one fetcher, schema versioning, incremental updates.
- **Backtest engine integration** — Optional integration with a mature engine (e.g. bt) or parity tests against a reference implementation for small universes.
- **Execution (paper/live)** - Target weights → broker orders with pre-trade risk checks; keep the same weight generator for backtest and live.
- **Packaging & DX** - Installable package (`pyproject.toml`), unified CLI, CI on `pytest`, tutorial notebook mirroring `run_all_real_data.py`.
- **Richer strategies** - ML/fundamental stock selection, timing overlays, multi-asset rotation with explicit regime logic.
- **Stronger reproducibility** — Run manifests (config hashes, data fingerprints, library versions, git commit) stored next to artifacts.

---

## Disclaimer

RL-PortfolioLab is provided for education and research only. Nothing in this repository constitutes investment, tax, or legal advice. Before trading or allocating real capital, consult qualified, licensed professionals. Historical and simulated performance may not reflect future outcomes.
