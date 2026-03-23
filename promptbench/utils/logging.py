"""
utils/logging.py — Logging configuration for promptbench.

Provides a single setup_logging() function that configures the root logger
for the promptbench namespace. Verbose (DEBUG) mode is enabled via the
--verbose CLI flag or by calling setup_logging(verbose=True).
"""

from __future__ import annotations

import logging
import sys


_HANDLER: logging.StreamHandler | None = None


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging for the 'promptbench' logger hierarchy.

    Args:
        verbose: If True, sets DEBUG level. Otherwise INFO.
    """
    global _HANDLER

    level = logging.DEBUG if verbose else logging.INFO

    root = logging.getLogger("promptbench")
    root.setLevel(level)

    if _HANDLER is None:
        _HANDLER = logging.StreamHandler(sys.stderr)
        _HANDLER.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root.addHandler(_HANDLER)
    else:
        _HANDLER.setLevel(level)
        root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger named 'promptbench.<name>'.

    Args:
        name: Sub-logger name (e.g. 'runner', 'optimizer').

    Returns:
        logging.Logger instance.
    """
    return logging.getLogger(f"promptbench.{name}")
