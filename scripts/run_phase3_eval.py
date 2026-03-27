from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    sys.path.insert(0, str(src_dir))


def main() -> None:
    _ensure_src_on_path()

    # Same preflight strategy as training: avoid hard-crash if numpy segfaults in this runtime.
    proc = subprocess.run(
        [sys.executable, "-c", "import numpy; print('numpy_ok')"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(
            "Phase 3 evaluation requires NumPy (SB3/Gymnasium dependency).\n"
            "In this runtime, `import numpy` fails.\n"
            "Run evaluation in a clean local environment where NumPy imports successfully.\n"
        )
        sys.exit(proc.returncode)

    from rl_portfoliolab.pipeline.phase3_eval import main as eval_main

    eval_main()


if __name__ == "__main__":
    main()

