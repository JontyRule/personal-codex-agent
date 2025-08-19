from __future__ import annotations

import logging
from typing import Optional


def get_logger(name: str = "personal_codex") -> logging.Logger:
    """Create or get a configured logger.

    - Logs INFO and above by default.
    - StreamHandler with simple format.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger
