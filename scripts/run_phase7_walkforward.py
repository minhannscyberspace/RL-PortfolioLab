from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    sys.path.insert(0, str(src_dir))


def main() -> None:
    _ensure_src_on_path()
    from rl_portfoliolab.pipeline.phase7_walkforward import main as phase7_main

    phase7_main()


if __name__ == "__main__":
    main()

