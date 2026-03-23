"""
optimization/optimizer.py — Automatic prompt optimization engine.

Steps:
1. Identify failing tests in a suite.
2. For each failing test, generate 3 candidate improved prompts using the LLM.
3. Re-run evaluation on each candidate.
4. Select the best-performing candidate.
5. Return structured OptimizationResult with improvement details.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from promptbench.config import SuiteConfig, TestCase, ExpectConfig
from promptbench.runner import TestResult

logger = logging.getLogger(__name__)

_OPTIMIZE_PROMPT_TEMPLATE = """\
You are an expert prompt engineer. A prompt is failing to produce the expected output.

## Original Prompt
{original_prompt}

## Expected Behavior
{expected_behavior}

## Actual Output (failing)
{actual_output}

## Failure Reason
{failure_reason}

Generate exactly 3 improved versions of the prompt that are more likely to produce the expected output.

Rules:
- Keep the same overall intent and task
- Fix the specific issue causing the failure
- Each variant should try a different approach

Respond with ONLY valid JSON in this exact format:
{{
  "variants": [
    {{"prompt": "<variant 1 text>"}},
    {{"prompt": "<variant 2 text>"}},
    {{"prompt": "<variant 3 text>"}}
  ]
}}
"""


@dataclass
class PromptVariant:
    """A candidate improved prompt."""

    prompt: str
    score: int = 0        # 0 = not evaluated; 1 = failed; 2 = passed
    test_result: Optional[TestResult] = None


@dataclass
class OptimizationResult:
    """Result of optimizing a single failing test."""

    test_name: str
    original_prompt: str
    original_failure: str
    variants: List[PromptVariant] = field(default_factory=list)
    best_prompt: Optional[str] = None
    improved: bool = False
    error: Optional[str] = None


class PromptOptimizer:
    """
    Automatically generates and evaluates improved prompt variants for failing tests.

    Usage:
        optimizer = PromptOptimizer(provider=my_provider, model="llama3-8b-8192")
        results = optimizer.optimize_suite(suite, failing_results)
    """

    NUM_VARIANTS = 3

    def __init__(self, provider, model: str) -> None:
        """
        Args:
            provider: An LLMProvider instance for generating variants.
            model:    Model identifier for both optimization and re-evaluation.
        """
        self._provider = provider
        self._model = model

    def optimize_suite(
        self,
        suite: SuiteConfig,
        failing_results: List[TestResult],
    ) -> List[OptimizationResult]:
        """
        Optimize all failing tests in *suite*.

        Args:
            suite:           The test suite configuration.
            failing_results: TestResult instances where passed=False.

        Returns:
            List of OptimizationResult, one per failing test.
        """
        results = []
        failing_map = {r.test_name: r for r in failing_results if not r.passed}

        for test in suite.tests:
            if test.name not in failing_map:
                continue
            failing_result = failing_map[test.name]
            logger.info("Optimizing prompt for test: %s", test.name)
            result = self.optimize_test(suite, test, failing_result)
            results.append(result)

        return results

    def optimize_test(
        self,
        suite: SuiteConfig,
        test: TestCase,
        failing_result: TestResult,
    ) -> OptimizationResult:
        """
        Generate improved prompt variants for a single failing test.

        Args:
            suite:          Suite configuration.
            test:           The failing test case.
            failing_result: The TestResult that failed.

        Returns:
            OptimizationResult with best variant selected.
        """
        original_prompt = test.input
        failure_reason = failing_result.failure_reason or "unknown failure"
        expected_behavior = _describe_expectations(test.expect)

        opt_result = OptimizationResult(
            test_name=test.name,
            original_prompt=original_prompt,
            original_failure=failure_reason,
        )

        # Generate candidate variants
        try:
            variants = self._generate_variants(
                original_prompt=original_prompt,
                expected_behavior=expected_behavior,
                actual_output=failing_result.actual_output,
                failure_reason=failure_reason,
            )
        except Exception as exc:
            opt_result.error = f"Failed to generate variants: {exc}"
            logger.error("Variant generation failed for '%s': %s", test.name, exc)
            return opt_result

        if not variants:
            opt_result.error = "LLM returned no valid variants."
            return opt_result

        opt_result.variants = variants

        # Re-evaluate each variant
        best_variant: Optional[PromptVariant] = None
        for variant in variants:
            try:
                passed = self._evaluate_variant(suite, test, variant.prompt)
                variant.score = 2 if passed else 1
                if passed and best_variant is None:
                    best_variant = variant
            except Exception as exc:
                logger.warning("Variant evaluation failed: %s", exc)
                variant.score = 0

        if best_variant is not None:
            opt_result.best_prompt = best_variant.prompt
            opt_result.improved = True
        else:
            # No variant passed — pick the one with highest score (all failed, pick first)
            opt_result.best_prompt = variants[0].prompt if variants else None
            opt_result.improved = False

        return opt_result

    def _generate_variants(
        self,
        original_prompt: str,
        expected_behavior: str,
        actual_output: str,
        failure_reason: str,
    ) -> List[PromptVariant]:
        """
        Ask the LLM to generate NUM_VARIANTS improved prompt candidates.

        Returns:
            List of PromptVariant instances (may be fewer than NUM_VARIANTS on parse error).
        """
        import json
        import re

        opt_prompt = _OPTIMIZE_PROMPT_TEMPLATE.format(
            original_prompt=original_prompt,
            expected_behavior=expected_behavior,
            actual_output=actual_output[:500],
            failure_reason=failure_reason[:200],
        )

        try:
            raw = self._provider.generate(opt_prompt, self._model)
        except Exception as exc:
            raise RuntimeError(f"LLM call failed: {exc}") from exc

        # Parse JSON response
        text = raw.strip()
        fence = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
        json_str = fence.group(1).strip() if fence else text
        brace = re.search(r"\{[\s\S]+\}", json_str)
        if brace:
            json_str = brace.group(0)

        try:
            data = json.loads(json_str)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Failed to parse optimizer response as JSON: %s", exc)
            return []

        raw_variants = data.get("variants", [])
        variants = []
        for v in raw_variants:
            if isinstance(v, dict) and isinstance(v.get("prompt"), str):
                variants.append(PromptVariant(prompt=v["prompt"].strip()))

        return variants[:self.NUM_VARIANTS]

    def _evaluate_variant(
        self,
        suite: SuiteConfig,
        test: TestCase,
        variant_prompt: str,
    ) -> bool:
        """
        Run the variant prompt against the LLM and check evaluators.

        Returns:
            True if the variant passes all evaluators.
        """
        from promptbench.evaluators.format_check import FormatChecker
        from promptbench.evaluators.length_check import LengthChecker
        from promptbench.evaluators.content_check import ContentChecker

        try:
            actual = self._provider.generate(variant_prompt, self._model)
        except Exception as exc:
            logger.warning("Variant LLM call failed: %s", exc)
            return False

        expect = test.expect
        checkers = [
            (FormatChecker().check, [actual, expect.format]),
            (LengthChecker().check, [actual, expect]),
            (ContentChecker().check, [actual, expect]),
        ]

        for checker_fn, args in checkers:
            try:
                passed, _ = checker_fn(*args)
                if not passed:
                    return False
            except Exception:
                pass

        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _describe_expectations(expect: ExpectConfig) -> str:
    """Convert an ExpectConfig to a human-readable description."""
    parts = []
    if expect.format:
        parts.append(f"Output format: {expect.format}")
    if expect.min_words:
        parts.append(f"Minimum {expect.min_words} words")
    if expect.max_words:
        parts.append(f"Maximum {expect.max_words} words")
    if expect.required_phrases:
        parts.append(f"Must include: {', '.join(repr(p) for p in expect.required_phrases)}")
    if expect.forbidden_phrases:
        parts.append(f"Must NOT include: {', '.join(repr(p) for p in expect.forbidden_phrases)}")
    if expect.tone:
        parts.append(f"Tone: {expect.tone}")
    return "; ".join(parts) if parts else "General quality and relevance"
