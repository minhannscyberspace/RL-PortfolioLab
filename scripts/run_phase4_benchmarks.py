from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    sys.path.insert(0, str(src_dir))


def main() -> None:
    _ensure_src_on_path()
    from rl_portfoliolab.pipeline.phase4_benchmarks import main as phase4_main

    phase4_main()


if __name__ == "__main__":
    main()

