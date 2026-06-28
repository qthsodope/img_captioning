from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str) -> dict[str, Any]:
    """Load a YAML config file."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("pyyaml is required to load config files.") from exc

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return data

