from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and torch when available."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:  # pragma: no cover - allows data tooling without torch
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

