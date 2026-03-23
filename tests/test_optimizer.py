"""
test_optimizer.py — Unit tests for the prompt optimization engine.

All LLM calls are mocked — no API key required.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from promptbench.config import SuiteConfig, TestCase, ExpectConfig
from promptbench.optimization.optimizer import PromptOptimizer, _describe_expectations
from promptbench.runner import TestResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_suite(name: str = "opt-suite") -> SuiteConfig:
    return SuiteConfig(
        name=name,
        model="llama3-8b-8192",
        api_provider="groq",
        tests=[
            TestCase(
                name="failing-test",
                input="What is the capital of France?",
                expect=ExpectConfig(required_phrases=["Paris"]),
            )
        ],
    )


def _make_failing_result() -> TestResult:
    return TestResult(
        test_name="failing-test",
        passed=False,
        actual_output="The capital is Lyon.",
        expected_output=None,
        duration_ms=100,
        failure_reason="Required phrase not found in output: 'Paris'",
    )


def _variants_response() -> str:
    return json.dumps({
        "variants": [
            {"prompt": "What is the capital city of France? Respond with just the city name."},
            {"prompt": "Name the capital of France. Include 'Paris' in your answer."},
            {"prompt": "Which city is the capital of France?"},
        ]
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPromptOptimizer:

    def _make_optimizer(self, generate_return: str = None) -> PromptOptimizer:
        provider = MagicMock()
        provider.generate.return_value = generate_return or _variants_response()
        return PromptOptimizer(provider=provider, model="llama3-8b-8192")

    def test_optimize_suite_returns_one_result_per_failure(self):
        optimizer = self._make_optimizer()
        suite = _make_suite()
        failing = [_make_failing_result()]

        results = optimizer.optimize_suite(suite, failing)

        assert len(results) == 1
        assert results[0].test_name == "failing-test"

    def test_optimize_skips_passing_tests(self):
        optimizer = self._make_optimizer()
        suite = _make_suite()
        passing_result = TestResult(
            test_name="failing-test",
            passed=True,
            actual_output="The capital is Paris.",
            expected_output="The capital is Paris.",
            duration_ms=100,
        )

        results = optimizer.optimize_suite(suite, [passing_result])

        assert len(results) == 0

    def test_variants_are_parsed_correctly(self):
        optimizer = self._make_optimizer()
        suite = _make_suite()
        failing = [_make_failing_result()]

        results = optimizer.optimize_suite(suite, failing)

        assert len(results[0].variants) == 3
        assert all(isinstance(v.prompt, str) for v in results[0].variants)

    def test_handles_invalid_json_response_gracefully(self):
        optimizer = self._make_optimizer(generate_return="This is not JSON.")
        suite = _make_suite()
        failing = [_make_failing_result()]

        results = optimizer.optimize_suite(suite, failing)

        # Should return a result with an error, not crash
        assert len(results) == 1
        result = results[0]
        assert result.error is not None or result.variants == []

    def test_handles_llm_call_failure_gracefully(self):
        provider = MagicMock()
        provider.generate.side_effect = RuntimeError("API down")
        optimizer = PromptOptimizer(provider=provider, model="llama3-8b-8192")
        suite = _make_suite()
        failing = [_make_failing_result()]

        results = optimizer.optimize_suite(suite, failing)

        assert len(results) == 1
        assert results[0].error is not None

    def test_improved_set_when_variant_passes(self):
        """When a variant passes all evaluators, improved=True and best_prompt is set."""
        # Provider returns variants on first call, then "Paris is the capital of France."
        # for the re-evaluation call
        call_count = [0]
        def side_effect(prompt, model):
            call_count[0] += 1
            if call_count[0] == 1:
                return _variants_response()
            return "Paris is the capital of France."

        provider = MagicMock()
        provider.generate.side_effect = side_effect
        optimizer = PromptOptimizer(provider=provider, model="llama3-8b-8192")
        suite = _make_suite()
        failing = [_make_failing_result()]

        results = optimizer.optimize_suite(suite, failing)

        result = results[0]
        assert result.improved is True
        assert result.best_prompt is not None


# ---------------------------------------------------------------------------
# _describe_expectations
# ---------------------------------------------------------------------------

class TestDescribeExpectations:

    def test_empty_expect(self):
        desc = _describe_expectations(ExpectConfig())
        assert "General quality" in desc

    def test_format_included(self):
        desc = _describe_expectations(ExpectConfig(format="bullet_points"))
        assert "bullet_points" in desc

    def test_required_phrases_included(self):
        desc = _describe_expectations(ExpectConfig(required_phrases=["Paris", "France"]))
        assert "Paris" in desc
        assert "France" in desc

    def test_word_limits_included(self):
        desc = _describe_expectations(ExpectConfig(min_words=10, max_words=50))
        assert "10" in desc
        assert "50" in desc
