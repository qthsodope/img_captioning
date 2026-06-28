from __future__ import annotations

import logging


def get_logger(name: str = "vicaption", level: int = logging.INFO) -> logging.Logger:
    """Create a simple console logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

