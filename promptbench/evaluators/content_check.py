"""
content_check.py — Verify that specific phrases and patterns appear (or don't appear)
in the LLM output.

Constraints supported (all optional):
  required_phrases    — strings that MUST appear verbatim (case-insensitive)
  forbidden_phrases   — strings that must NOT appear (case-insensitive)
  forbidden_patterns  — regex patterns that must NOT match
  required_patterns   — regex patterns that MUST match
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from promptbench.config import ExpectConfig


class ContentChecker:
    """
    Evaluator that checks for required/forbidden phrases and regex patterns
    in the LLM output.
    """

    def check(self, output: str, expect: ExpectConfig) -> Tuple[bool, Optional[str]]:
        """
        Evaluate all content constraints for *output*.

        Args:
            output: The raw string produced by the LLM.
            expect: The ExpectConfig for this test.

        Returns:
            (passed, failure_reason) — failure_reason is None when passed is True.
        """
        failures: List[str] = []
        lower_output = output.lower()

        # Required phrases — must be present
        for phrase in expect.required_phrases:
            if phrase.lower() not in lower_output:
                failures.append(
                    f"Required phrase not found in output: '{phrase}'"
                )

        # Forbidden phrases — must not be present
        for phrase in expect.forbidden_phrases:
            if phrase.lower() in lower_output:
                failures.append(
                    f"Forbidden phrase found in output: '{phrase}'"
                )

        # Required patterns — regex must match
        for pattern_str in expect.required_patterns:
            try:
                pattern = re.compile(pattern_str, re.IGNORECASE | re.MULTILINE)
            except re.error as exc:
                failures.append(
                    f"Invalid required_pattern regex '{pattern_str}': {exc}"
                )
                continue
            if not pattern.search(output):
                failures.append(
                    f"Required pattern not matched in output: r'{pattern_str}'"
                )

        # Forbidden patterns — regex must NOT match
        for pattern_str in expect.forbidden_patterns:
            try:
                pattern = re.compile(pattern_str, re.IGNORECASE | re.MULTILINE)
            except re.error as exc:
                failures.append(
                    f"Invalid forbidden_pattern regex '{pattern_str}': {exc}"
                )
                continue
            match = pattern.search(output)
            if match:
                failures.append(
                    f"Forbidden pattern matched in output: r'{pattern_str}' "
                    f"(matched: '{match.group(0)}')"
                )

        if failures:
            return False, " | ".join(failures)
        return True, None

    # ------------------------------------------------------------------
    # Utility methods (exposed for testing)
    # ------------------------------------------------------------------

    @staticmethod
    def contains_phrase(output: str, phrase: str, case_sensitive: bool = False) -> bool:
        """Return True if *phrase* appears in *output*."""
        if case_sensitive:
            return phrase in output
        return phrase.lower() in output.lower()

    @staticmethod
    def matches_pattern(output: str, pattern: str) -> bool:
        """Return True if the regex *pattern* matches anywhere in *output*."""
        return bool(re.search(pattern, output, re.IGNORECASE | re.MULTILINE))
