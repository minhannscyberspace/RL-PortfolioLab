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
    # Fail fast with a clear error if NumPy segfaults on import in this runtime.
    # We do this check in a subprocess to avoid crashing this parent process.
    proc = subprocess.run(
        [sys.executable, "-c", "import numpy; print('numpy_ok')"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(
            "Phase 3 requires NumPy + PyTorch (Stable-Baselines3).\n"
            "In this runtime, `import numpy` fails (often as a segfault).\n"
            "Fix by using a clean Python environment where NumPy imports successfully, then install:\n"
            "  pip install gymnasium stable-baselines3 torch\n"
            "Then re-run this script.\n"
        )
        sys.exit(proc.returncode)

    from rl_portfoliolab.pipeline.phase3_train import main as phase3_main

    phase3_main()


if __name__ == "__main__":
    main()

