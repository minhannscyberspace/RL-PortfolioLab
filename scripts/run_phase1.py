from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    sys.path.insert(0, str(src_dir))


def main() -> None:
    _ensure_src_on_path()

    from rl_portfoliolab.pipeline.phase1 import main as pipeline_main

    # Delegate to the pipeline entrypoint.
    # We keep this wrapper so users can run `python scripts/run_phase1.py ...`
    pipeline_main()


if __name__ == "__main__":
    main()

