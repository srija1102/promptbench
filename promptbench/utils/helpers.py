"""
utils/helpers.py — General-purpose helper functions for promptbench.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")


def truncate(text: str, max_chars: int = 200, suffix: str = "…") -> str:
    """
    Truncate *text* to at most *max_chars* characters, appending *suffix*.

    Args:
        text:      Input string.
        max_chars: Maximum length (excluding suffix).
        suffix:    Appended when truncation occurs.

    Returns:
        Possibly-truncated string.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + suffix


def slugify(text: str) -> str:
    """
    Convert *text* to a filesystem-safe slug.

    Args:
        text: Any string (e.g. suite name).

    Returns:
        Lowercase alphanumeric slug with hyphens instead of spaces/special chars.

    Example:
        >>> slugify("My Suite: v2!")
        'my-suite-v2'
    """
    slug = text.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug or "unnamed"


def retry(
    fn: Callable[[], T],
    max_attempts: int = 3,
    base_delay: float = 1.0,
    exceptions: tuple = (Exception,),
) -> T:
    """
    Retry *fn* up to *max_attempts* times with exponential backoff.

    Args:
        fn:           Zero-argument callable.
        max_attempts: Maximum number of attempts.
        base_delay:   Initial sleep in seconds (doubles each retry).
        exceptions:   Tuple of exception types to catch and retry on.

    Returns:
        Return value of *fn* on success.

    Raises:
        The last exception if all attempts fail.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except exceptions as exc:  # type: ignore[misc]
            last_exc = exc
            if attempt < max_attempts:
                time.sleep(base_delay * (2 ** (attempt - 1)))
    raise last_exc  # type: ignore[misc]


def safe_json_loads(text: str) -> Optional[Any]:
    """
    Attempt to parse *text* as JSON, returning None on failure.

    Strips markdown code fences before parsing.

    Args:
        text: String potentially containing JSON.

    Returns:
        Parsed object or None.
    """
    import json

    stripped = text.strip()
    # Strip ```json ... ``` fences
    fence = re.search(r"```(?:json)?\s*([\s\S]+?)```", stripped)
    if fence:
        stripped = fence.group(1).strip()
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None


def word_count(text: str) -> int:
    """Return the number of whitespace-separated words in *text*."""
    return len(text.split())


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to the range [lo, hi]."""
    return max(lo, min(hi, value))
