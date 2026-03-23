"""
test_llm_judge.py — Unit tests for the LLM-as-judge evaluator.

All LLM calls are mocked — no API key required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from promptbench.evaluators.llm_judge import LLMJudge, LLMJudgeResult, _parse_judge_response


# ===========================================================================
# _parse_judge_response
# ===========================================================================

class TestParseJudgeResponse:
    """Tests for the JSON parsing helper."""

    def test_parses_valid_json(self):
        raw = '{"score": 4, "reason": "Good response, mostly correct."}'
        result = _parse_judge_response(raw)
        assert result is not None
        score, reason = result
        assert score == 4
        assert "Good response" in reason

    def test_parses_score_5(self):
        raw = '{"score": 5, "reason": "Perfect."}'
        result = _parse_judge_response(raw)
        assert result is not None
        assert result[0] == 5

    def test_parses_score_1(self):
        raw = '{"score": 1, "reason": "Completely wrong."}'
        result = _parse_judge_response(raw)
        assert result is not None
        assert result[0] == 1

    def test_parses_json_in_markdown_fence(self):
        raw = 'Here is my evaluation:\n```json\n{"score": 3, "reason": "Acceptable."}\n```'
        result = _parse_judge_response(raw)
        assert result is not None
        assert result[0] == 3

    def test_parses_json_embedded_in_prose(self):
        raw = 'My evaluation: {"score": 4, "reason": "Good."} End.'
        result = _parse_judge_response(raw)
        assert result is not None
        assert result[0] == 4

    def test_returns_none_for_invalid_json(self):
        result = _parse_judge_response("This is not JSON at all.")
        assert result is None

    def test_returns_none_for_score_out_of_range(self):
        raw = '{"score": 6, "reason": "Invalid score."}'
        result = _parse_judge_response(raw)
        assert result is None

    def test_returns_none_for_missing_score(self):
        raw = '{"reason": "No score field."}'
        result = _parse_judge_response(raw)
        assert result is None

    def test_returns_none_for_non_numeric_score(self):
        raw = '{"score": "four", "reason": "Wrong type."}'
        result = _parse_judge_response(raw)
        assert result is None

    def test_handles_float_score(self):
        raw = '{"score": 3.0, "reason": "Float score is fine."}'
        result = _parse_judge_response(raw)
        assert result is not None
        assert result[0] == 3


# ===========================================================================
# LLMJudge.evaluate
# ===========================================================================

class TestLLMJudge:
    """Tests for LLMJudge with a mocked provider."""

    def _make_judge(self, mock_response: str, threshold: int = 3) -> LLMJudge:
        provider = MagicMock()
        provider.generate.return_value = mock_response
        return LLMJudge(provider=provider, model="test-model", pass_threshold=threshold)

    def test_high_score_passes(self):
        judge = self._make_judge('{"score": 5, "reason": "Excellent."}')
        result = judge.evaluate("What is 2+2?", "Should return 4", "4")
        assert result.passed is True
        assert result.score == 5
        assert "Excellent" in result.reason

    def test_low_score_fails(self):
        judge = self._make_judge('{"score": 1, "reason": "Completely wrong."}')
        result = judge.evaluate("What is 2+2?", "Should return 4", "fish")
        assert result.passed is False
        assert result.score == 1

    def test_at_threshold_passes(self):
        judge = self._make_judge('{"score": 3, "reason": "Acceptable."}', threshold=3)
        result = judge.evaluate("Test?", "Something", "response")
        assert result.passed is True

    def test_below_threshold_fails(self):
        judge = self._make_judge('{"score": 2, "reason": "Poor."}', threshold=3)
        result = judge.evaluate("Test?", "Something", "response")
        assert result.passed is False

    def test_parse_failure_retries_and_fails_safely(self):
        """When parse fails twice, returns score=1 and passed=False."""
        provider = MagicMock()
        provider.generate.return_value = "This is not valid JSON."
        judge = LLMJudge(provider=provider, model="test-model")
        result = judge.evaluate("Test?", "Something", "response")
        assert result.passed is False
        assert result.score == 1
        assert "parse" in result.reason.lower() or "Failed" in result.reason

    def test_llm_call_failure_returns_failing_result(self):
        """If the LLM call throws, returns a failing result."""
        provider = MagicMock()
        provider.generate.side_effect = ConnectionError("Network error")
        judge = LLMJudge(provider=provider, model="test-model")
        result = judge.evaluate("Test?", "Something", "response")
        assert result.passed is False

    def test_check_interface_returns_tuple(self):
        judge = self._make_judge('{"score": 5, "reason": "Perfect."}')
        passed, reason = judge.check("Q?", "Expected behavior", "Great answer.")
        assert passed is True
        assert reason is None

    def test_check_interface_failure_includes_score(self):
        judge = self._make_judge('{"score": 1, "reason": "Wrong."}')
        passed, reason = judge.check("Q?", "Expected behavior", "Bad answer.")
        assert passed is False
        assert "1/5" in reason or "score" in reason.lower()

    def test_raw_response_stored(self):
        raw = '{"score": 4, "reason": "Good."}'
        judge = self._make_judge(raw)
        result = judge.evaluate("Q?", "Something", "answer")
        assert result.raw_response == raw
