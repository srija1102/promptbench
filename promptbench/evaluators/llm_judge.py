"""
llm_judge.py — Use an LLM as a quality judge to score outputs on a 1–5 scale.

Steps:
1. Build an evaluation prompt from the test input, expected behavior, and actual output.
2. Call the configured LLM provider.
3. Parse the JSON response safely (score + reason).
4. Retry once if parsing fails, then fail safely.

Returns:
    LLMJudgeResult(score=1–5, reason="...", passed=bool)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Minimum score to be considered passing (configurable per test)
DEFAULT_PASS_THRESHOLD = 3


_JUDGE_PROMPT_TEMPLATE = """\
You are an expert AI output quality evaluator.

Your job is to score the following AI response on a scale of 1 to 5.

## Test Input
{test_input}

## Expected Behavior
{expected_behavior}

## Actual Output
{actual_output}

## Scoring Criteria
- 5: Excellent — fully satisfies requirements, clear, accurate, well-formatted
- 4: Good — mostly satisfies requirements with minor issues
- 3: Acceptable — partially satisfies requirements; some issues present
- 2: Poor — significant issues; key requirements not met
- 1: Failing — wrong, incoherent, or completely misses the requirements

## Response Format (STRICT — respond ONLY with valid JSON, no other text)
{{
  "score": <integer 1-5>,
  "reason": "<one to two sentence explanation>"
}}
"""


@dataclass
class LLMJudgeResult:
    """Result of an LLM-as-judge evaluation."""

    score: int          # 1–5
    reason: str
    passed: bool        # True if score >= pass_threshold
    raw_response: str   # The raw LLM output (for debugging)


def _parse_judge_response(raw: str) -> Optional[Tuple[int, str]]:
    """
    Extract (score, reason) from the judge LLM's raw response.

    Handles:
    - Pure JSON responses
    - JSON embedded in markdown code fences
    - JSON embedded in surrounding prose

    Args:
        raw: The raw string returned by the judge LLM.

    Returns:
        (score, reason) tuple, or None if parsing failed.
    """
    text = raw.strip()

    # Try to find JSON block (possibly inside fences)
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    json_str = fence_match.group(1).strip() if fence_match else text

    # If no fence, try to find the first {...} block
    if not fence_match:
        brace_match = re.search(r"\{[\s\S]+\}", text)
        if brace_match:
            json_str = brace_match.group(0)

    try:
        data = json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(data, dict):
        return None

    score = data.get("score")
    reason = data.get("reason", "")

    if not isinstance(score, (int, float)):
        return None
    score = int(score)
    if not (1 <= score <= 5):
        return None

    return score, str(reason).strip()


class LLMJudge:
    """
    Evaluator that calls an LLM to score output quality on a 1–5 scale.

    Usage:
        judge = LLMJudge(provider=my_provider, model="llama3-8b-8192")
        result = judge.evaluate(test_input="...", expected="...", actual="...")
    """

    def __init__(
        self,
        provider,
        model: str,
        pass_threshold: int = DEFAULT_PASS_THRESHOLD,
    ) -> None:
        """
        Args:
            provider:       An LLMProvider instance used to call the judge model.
            model:          Model identifier to use for judging.
            pass_threshold: Minimum score (inclusive) to mark as passed (default 3).
        """
        self._provider = provider
        self._model = model
        self._pass_threshold = pass_threshold

    def evaluate(
        self,
        test_input: str,
        expected_behavior: str,
        actual_output: str,
    ) -> LLMJudgeResult:
        """
        Score *actual_output* on a 1–5 scale using an LLM judge.

        Args:
            test_input:          The original prompt / user input.
            expected_behavior:   Human-readable description of what good output looks like.
            actual_output:       The LLM response being evaluated.

        Returns:
            LLMJudgeResult with score, reason, and passed flag.
        """
        judge_prompt = _JUDGE_PROMPT_TEMPLATE.format(
            test_input=test_input,
            expected_behavior=expected_behavior,
            actual_output=actual_output,
        )

        raw = self._call_with_retry(judge_prompt)
        parsed = _parse_judge_response(raw)

        if parsed is None:
            logger.warning("Judge response parse failed on first attempt, retrying...")
            raw = self._call_with_retry(judge_prompt)
            parsed = _parse_judge_response(raw)

        if parsed is None:
            logger.error("Judge response parse failed after retry. Raw: %s", raw[:200])
            return LLMJudgeResult(
                score=1,
                reason="Failed to parse judge response — treating as failure.",
                passed=False,
                raw_response=raw,
            )

        score, reason = parsed
        return LLMJudgeResult(
            score=score,
            reason=reason,
            passed=score >= self._pass_threshold,
            raw_response=raw,
        )

    def check(
        self,
        test_input: str,
        expected_behavior: str,
        actual_output: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Evaluator interface compatible with the runner's evaluator pattern.

        Returns:
            (passed, failure_reason) — failure_reason is None when passed.
        """
        result = self.evaluate(test_input, expected_behavior, actual_output)
        if result.passed:
            return True, None
        return False, (
            f"LLM judge score {result.score}/5 (threshold {self._pass_threshold}): "
            f"{result.reason}"
        )

    def _call_with_retry(self, prompt: str) -> str:
        """Call the provider, swallowing transient errors and returning empty string."""
        try:
            return self._provider.generate(prompt, self._model)
        except Exception as exc:
            logger.error("Judge LLM call failed: %s", exc)
            return ""
