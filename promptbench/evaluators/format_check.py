"""
format_check.py — Verify that an LLM output matches an expected structural format.

Supported formats:
  bullet_points   — lines starting with -, *, or •
  json            — valid parseable JSON
  numbered_list   — lines starting with 1. 2. 3.
  markdown        — contains markdown syntax elements
  plain_text      — no special formatting characters detected
"""

from __future__ import annotations

import json
import re
from typing import Optional, Tuple


# Regex patterns used for format detection
_BULLET_LINE = re.compile(r"^\s*[-*•]\s+", re.MULTILINE)
_NUMBERED_LINE = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)
_MARKDOWN_PATTERNS = [
    re.compile(r"#{1,6}\s"),          # headings
    re.compile(r"\*\*[^*]+\*\*"),     # bold
    re.compile(r"\*[^*]+\*"),         # italic
    re.compile(r"`[^`]+`"),           # inline code
    re.compile(r"```"),               # code block
    re.compile(r"^\s*>\s+", re.MULTILINE),  # blockquote
    re.compile(r"\[.+\]\(.+\)"),      # link
]
_SPECIAL_CHARS = re.compile(r"[#*`>|\[\]_~\\]")


class FormatChecker:
    """
    Evaluator that checks whether an output string matches a declared format.

    Supported format values:
        "bullet_points", "json", "numbered_list", "markdown", "plain_text"
    """

    SUPPORTED_FORMATS = {
        "bullet_points",
        "json",
        "numbered_list",
        "markdown",
        "plain_text",
    }

    def check(self, output: str, expected_format: Optional[str]) -> Tuple[bool, Optional[str]]:
        """
        Evaluate whether *output* conforms to *expected_format*.

        Args:
            output: The raw string produced by the LLM.
            expected_format: One of the supported format strings, or None to skip.

        Returns:
            (passed, failure_reason) — failure_reason is None when passed is True.

        Raises:
            ValueError: If expected_format is not a known format.
        """
        if expected_format is None:
            return True, None

        fmt = expected_format.lower().strip()
        if fmt not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unknown format '{expected_format}'. "
                f"Valid options: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
            )

        if fmt == "bullet_points":
            return self._check_bullet_points(output)
        if fmt == "json":
            return self._check_json(output)
        if fmt == "numbered_list":
            return self._check_numbered_list(output)
        if fmt == "markdown":
            return self._check_markdown(output)
        if fmt == "plain_text":
            return self._check_plain_text(output)

        return True, None  # unreachable but satisfies type checker

    # ------------------------------------------------------------------
    # Private checkers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_bullet_points(output: str) -> Tuple[bool, Optional[str]]:
        """Require at least one line starting with -, *, or •."""
        if _BULLET_LINE.search(output):
            return True, None
        return (
            False,
            "Expected bullet-point list (lines starting with -, *, or •) "
            "but none were found in the output.",
        )

    @staticmethod
    def _check_json(output: str) -> Tuple[bool, Optional[str]]:
        """Require that the output is valid JSON (object or array)."""
        stripped = output.strip()
        # Try to extract JSON block from markdown code fences
        fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)```", stripped)
        json_str = fence_match.group(1).strip() if fence_match else stripped
        try:
            json.loads(json_str)
            return True, None
        except json.JSONDecodeError as exc:
            return False, f"Expected valid JSON output but got a parse error: {exc}"

    @staticmethod
    def _check_numbered_list(output: str) -> Tuple[bool, Optional[str]]:
        """Require at least one line starting with a number followed by a period."""
        if _NUMBERED_LINE.search(output):
            return True, None
        return (
            False,
            "Expected numbered list (lines starting with 1. 2. 3.) "
            "but none were found in the output.",
        )

    @staticmethod
    def _check_markdown(output: str) -> Tuple[bool, Optional[str]]:
        """Require at least one recognizable markdown syntax element."""
        for pattern in _MARKDOWN_PATTERNS:
            if pattern.search(output):
                return True, None
        return (
            False,
            "Expected markdown formatting (headings, bold, italic, code blocks, links) "
            "but no markdown syntax was detected.",
        )

    @staticmethod
    def _check_plain_text(output: str) -> Tuple[bool, Optional[str]]:
        """Require that the output contains no markdown-like special characters."""
        if _SPECIAL_CHARS.search(output):
            return (
                False,
                "Expected plain text output with no special formatting, "
                "but found markdown-like characters (e.g. #, *, `, >).",
            )
        return True, None
