"""
tone_check.py — Rule-based tone and style evaluation (no external API required).

Detects:
  - Excessive hedging: "I think", "maybe", "possibly", "I believe", etc.
  - Refusal patterns: "I cannot", "I'm unable", "As an AI", etc.
  - Overly casual language: "hey", "yeah", "gonna", "wanna", etc.

When a baseline output is available, violations are flagged only if they
appear in the current output but were absent from the baseline. When
comparing fresh against a declared tone, the rules are applied absolutely.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Pattern catalogs
# ---------------------------------------------------------------------------

_HEDGING_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bi\s+think\b", re.IGNORECASE),
    re.compile(r"\bi\s+believe\b", re.IGNORECASE),
    re.compile(r"\bmaybe\b", re.IGNORECASE),
    re.compile(r"\bperhaps\b", re.IGNORECASE),
    re.compile(r"\bpossibly\b", re.IGNORECASE),
    re.compile(r"\bprobably\b", re.IGNORECASE),
    re.compile(r"\bit\s+seems\b", re.IGNORECASE),
    re.compile(r"\bseems?\s+like\b", re.IGNORECASE),
    re.compile(r"\bI'm not sure\b", re.IGNORECASE),
    re.compile(r"\bit's?\s+possible\b", re.IGNORECASE),
]

_REFUSAL_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bI\s+cannot\b", re.IGNORECASE),
    re.compile(r"\bI\s+can't\b", re.IGNORECASE),
    re.compile(r"\bI\s*'m\s+unable\b", re.IGNORECASE),
    re.compile(r"\bAs\s+an\s+AI\b", re.IGNORECASE),
    re.compile(r"\bAs\s+a\s+language\s+model\b", re.IGNORECASE),
    re.compile(r"\bI\s+don't\s+have\s+(the\s+)?ability\b", re.IGNORECASE),
    re.compile(r"\bI\s+am\s+not\s+able\b", re.IGNORECASE),
    re.compile(r"\bI\s+must\s+decline\b", re.IGNORECASE),
    re.compile(r"\bI\s+apologize\b", re.IGNORECASE),
    re.compile(r"\bI\s+cannot\s+provide\b", re.IGNORECASE),
]

_CASUAL_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bhey\b", re.IGNORECASE),
    re.compile(r"\byeah\b", re.IGNORECASE),
    re.compile(r"\bgonna\b", re.IGNORECASE),
    re.compile(r"\bwanna\b", re.IGNORECASE),
    re.compile(r"\bgotta\b", re.IGNORECASE),
    re.compile(r"\bnope\b", re.IGNORECASE),
    re.compile(r"\byup\b", re.IGNORECASE),
    re.compile(r"\bcool\b", re.IGNORECASE),
    re.compile(r"\bawesome\b", re.IGNORECASE),
    re.compile(r"\bkinda\b", re.IGNORECASE),
    re.compile(r"\bsorta\b", re.IGNORECASE),
    re.compile(r"\blol\b", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

@dataclass
class ToneProfile:
    """Describes which tone categories are present in a piece of text."""

    has_hedging: bool = False
    has_refusals: bool = False
    has_casual: bool = False
    hedging_examples: List[str] = field(default_factory=list)
    refusal_examples: List[str] = field(default_factory=list)
    casual_examples: List[str] = field(default_factory=list)


def _find_matches(text: str, patterns: List[re.Pattern]) -> List[str]:
    """Return a de-duplicated list of matched substrings from *text*."""
    seen = set()
    results = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            token = match.group(0).lower()
            if token not in seen:
                seen.add(token)
                results.append(match.group(0))
    return results


def analyze_tone(text: str) -> ToneProfile:
    """
    Analyze *text* and return a ToneProfile indicating which tone
    categories are present.

    Args:
        text: Any string (LLM output or baseline).

    Returns:
        ToneProfile instance.
    """
    hedging = _find_matches(text, _HEDGING_PATTERNS)
    refusals = _find_matches(text, _REFUSAL_PATTERNS)
    casual = _find_matches(text, _CASUAL_PATTERNS)
    return ToneProfile(
        has_hedging=bool(hedging),
        has_refusals=bool(refusals),
        has_casual=bool(casual),
        hedging_examples=hedging,
        refusal_examples=refusals,
        casual_examples=casual,
    )


# ---------------------------------------------------------------------------
# Checker class
# ---------------------------------------------------------------------------

_TONE_RULES: Dict[str, Dict[str, bool]] = {
    "professional": {"hedging": False, "refusals": False, "casual": False},
    "casual":       {"hedging": True,  "refusals": False, "casual": True},
    "neutral":      {"hedging": False, "refusals": False, "casual": False},
}


class ToneChecker:
    """
    Rule-based tone evaluator.

    Two modes of operation:
      1. Absolute tone check — compare against declared tone rules.
      2. Baseline drift check — flag new violations not present in baseline.
    """

    def check(
        self,
        output: str,
        tone: Optional[str],
        baseline_output: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Evaluate tone of *output*.

        Args:
            output: The current LLM output to evaluate.
            tone: Optional declared tone ('professional', 'casual', 'neutral').
            baseline_output: Optional previous output to compare drift against.

        Returns:
            (passed, failure_reason).
        """
        if tone is None and baseline_output is None:
            return True, None

        failures: List[str] = []
        current_profile = analyze_tone(output)

        if tone is not None:
            tone = tone.lower().strip()
            if tone not in _TONE_RULES:
                return False, (
                    f"Unknown tone '{tone}'. Valid options: "
                    f"{', '.join(sorted(_TONE_RULES))}"
                )
            rules = _TONE_RULES[tone]

            if not rules["hedging"] and current_profile.has_hedging:
                examples = ", ".join(f"'{e}'" for e in current_profile.hedging_examples[:3])
                failures.append(
                    f"Tone '{tone}' should not contain hedging language "
                    f"(found: {examples})."
                )
            if not rules["refusals"] and current_profile.has_refusals:
                examples = ", ".join(f"'{e}'" for e in current_profile.refusal_examples[:3])
                failures.append(
                    f"Tone '{tone}' should not contain refusal patterns "
                    f"(found: {examples})."
                )
            if not rules["casual"] and current_profile.has_casual:
                examples = ", ".join(f"'{e}'" for e in current_profile.casual_examples[:3])
                failures.append(
                    f"Tone '{tone}' should not contain casual language "
                    f"(found: {examples})."
                )

        elif baseline_output is not None:
            # Drift mode: flag categories that appear now but didn't in baseline
            baseline_profile = analyze_tone(baseline_output)

            if current_profile.has_hedging and not baseline_profile.has_hedging:
                examples = ", ".join(f"'{e}'" for e in current_profile.hedging_examples[:3])
                failures.append(
                    f"Hedging language appeared that was absent in baseline "
                    f"(found: {examples})."
                )
            if current_profile.has_refusals and not baseline_profile.has_refusals:
                examples = ", ".join(f"'{e}'" for e in current_profile.refusal_examples[:3])
                failures.append(
                    f"Refusal patterns appeared that were absent in baseline "
                    f"(found: {examples})."
                )
            if current_profile.has_casual and not baseline_profile.has_casual:
                examples = ", ".join(f"'{e}'" for e in current_profile.casual_examples[:3])
                failures.append(
                    f"Casual language appeared that was absent in baseline "
                    f"(found: {examples})."
                )

        if failures:
            return False, " | ".join(failures)
        return True, None

    @staticmethod
    def analyze(text: str) -> ToneProfile:
        """Public wrapper around analyze_tone for external use."""
        return analyze_tone(text)
