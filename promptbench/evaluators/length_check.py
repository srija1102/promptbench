"""
length_check.py — Enforce word count, bullet count, and sentence count constraints.

Constraints supported (all optional, any combination):
  max_words       — upper bound on total word count
  min_words       — lower bound on total word count
  max_bullets     — upper bound on bullet/list item count
  min_bullets     — lower bound on bullet/list item count
  max_sentences   — upper bound on sentence count
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from promptbench.config import ExpectConfig


_BULLET_LINE = re.compile(r"^\s*[-*•]\s+|\s*\d+\.\s+", re.MULTILINE)
# Sentence boundary: ends with . ? ! followed by whitespace or end of string
_SENTENCE_END = re.compile(r"[.?!]+(?:\s|$)")


def _count_words(text: str) -> int:
    """Return the number of whitespace-delimited words in *text*."""
    return len(text.split())


def _count_bullets(text: str) -> int:
    """Return the number of bullet or numbered-list lines in *text*."""
    return len(_BULLET_LINE.findall(text))


def _count_sentences(text: str) -> int:
    """
    Return an approximate sentence count.

    Uses punctuation boundaries (.?!) as heuristic; avoids counting
    abbreviations like 'e.g.' as sentence breaks.
    """
    # Remove common abbreviations to avoid false sentence splits
    cleaned = re.sub(r"\b(e\.g|i\.e|etc|vs|Mr|Ms|Dr|Prof)\.", "", text)
    matches = _SENTENCE_END.findall(cleaned)
    # If no sentence-ending punctuation found, count as 1 sentence if non-empty
    if not matches and text.strip():
        return 1
    return len(matches)


class LengthChecker:
    """
    Evaluator that checks word count, bullet count, and sentence count against
    the constraints declared in an ExpectConfig.
    """

    def check(self, output: str, expect: ExpectConfig) -> Tuple[bool, Optional[str]]:
        """
        Evaluate all length constraints for *output*.

        Args:
            output: The raw string produced by the LLM.
            expect: The ExpectConfig for this test (constraints read from YAML).

        Returns:
            (passed, failure_reason) — failure_reason is None when passed is True.
            If multiple constraints fail, all failures are combined in the reason.
        """
        failures: List[str] = []

        word_count = _count_words(output)
        bullet_count = _count_bullets(output)
        sentence_count = _count_sentences(output)

        if expect.max_words is not None and word_count > expect.max_words:
            failures.append(
                f"Output has {word_count} words, exceeds max_words={expect.max_words}."
            )
        if expect.min_words is not None and word_count < expect.min_words:
            failures.append(
                f"Output has {word_count} words, below min_words={expect.min_words}."
            )
        if expect.max_bullets is not None and bullet_count > expect.max_bullets:
            failures.append(
                f"Output has {bullet_count} bullets, exceeds max_bullets={expect.max_bullets}."
            )
        if expect.min_bullets is not None and bullet_count < expect.min_bullets:
            failures.append(
                f"Output has {bullet_count} bullets, below min_bullets={expect.min_bullets}."
            )
        if expect.max_sentences is not None and sentence_count > expect.max_sentences:
            failures.append(
                f"Output has {sentence_count} sentences, "
                f"exceeds max_sentences={expect.max_sentences}."
            )

        if failures:
            return False, " | ".join(failures)
        return True, None

    # ------------------------------------------------------------------
    # Utility (exposed for testing / debugging)
    # ------------------------------------------------------------------

    @staticmethod
    def word_count(text: str) -> int:
        """Return the word count of *text*."""
        return _count_words(text)

    @staticmethod
    def bullet_count(text: str) -> int:
        """Return the bullet/numbered-list item count of *text*."""
        return _count_bullets(text)

    @staticmethod
    def sentence_count(text: str) -> int:
        """Return the approximate sentence count of *text*."""
        return _count_sentences(text)
