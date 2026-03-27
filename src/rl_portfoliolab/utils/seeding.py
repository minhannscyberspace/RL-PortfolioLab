from __future__ import annotations

import random


def set_seed(seed: int) -> None:
    """Set deterministic seeds for python RNG."""
    random.seed(seed)

