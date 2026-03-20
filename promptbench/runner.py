"""
runner.py — Core test execution engine.

Loads a YAML suite, calls the LLM for each test, runs all enabled evaluators,
and returns structured TestResult instances. Also handles baseline saving.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from promptbench.config import SuiteConfig, TestCase, get_api_key
from promptbench.evaluators.format_check import FormatChecker
from promptbench.evaluators.length_check import LengthChecker
from promptbench.evaluators.content_check import ContentChecker
from promptbench.evaluators.tone_check import ToneChecker
from promptbench.evaluators.semantic_check import SemanticChecker
from promptbench.storage import Storage, ResultRecord


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    """
    Structured result for a single test case execution.

    Attributes:
        test_name:      Identifier of the test case.
        passed:         True if all evaluators passed.
        actual_output:  The raw string returned by the LLM.
        expected_output: The baseline output (if available), else None.
        duration_ms:    Wall-clock time spent on this test, in milliseconds.
        failure_reason: Human-readable summary of all failures, or None.
    """

    test_name: str
    passed: bool
    actual_output: str
    expected_output: Optional[str]
    duration_ms: int
    failure_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# LLM clients
# ---------------------------------------------------------------------------

def _call_groq(prompt: str, model: str, api_key: str) -> str:
    """
    Call the Groq chat completions API.

    Args:
        prompt:  The full user message / prompt string.
        model:   Model identifier (e.g. "llama3-8b-8192").
        api_key: Groq API key.

    Returns:
        The model's text response.

    Raises:
        requests.HTTPError: On non-2xx responses.
        RuntimeError:       If the response JSON is malformed.
    """
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    response = requests.post(url, json=payload, headers=headers, timeout=60)
    if not response.ok:
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        raise RuntimeError(
            f"Groq API error {response.status_code}: {detail}"
        )
    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(
            f"Unexpected Groq API response format: {data}"
        ) from exc


def _call_openai(prompt: str, model: str, api_key: str) -> str:
    """
    Call the OpenAI chat completions API.

    Args:
        prompt:  The full user message / prompt string.
        model:   Model identifier (e.g. "gpt-4o-mini").
        api_key: OpenAI API key.

    Returns:
        The model's text response.
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    response = requests.post(url, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(
            f"Unexpected OpenAI API response format: {data}"
        ) from exc


def _call_anthropic(prompt: str, model: str, api_key: str) -> str:
    """
    Call the Anthropic Messages API.

    Args:
        prompt:  The full user message / prompt string.
        model:   Model identifier (e.g. "claude-3-haiku-20240307").
        api_key: Anthropic API key.

    Returns:
        The model's text response.
    """
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": prompt}],
    }
    response = requests.post(url, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    data = response.json()
    try:
        return data["content"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(
            f"Unexpected Anthropic API response format: {data}"
        ) from exc


def call_llm(prompt: str, model: str, provider: str, api_key: str) -> str:
    """
    Dispatch to the correct LLM client based on *provider*.

    Args:
        prompt:   The full prompt to send.
        model:    Model identifier.
        provider: One of 'groq', 'openai', 'anthropic'.
        api_key:  API key for the provider.

    Returns:
        The model's text response.
    """
    if provider == "groq":
        return _call_groq(prompt, model, api_key)
    if provider == "openai":
        return _call_openai(prompt, model, api_key)
    if provider == "anthropic":
        return _call_anthropic(prompt, model, api_key)
    raise ValueError(
        f"Unknown provider '{provider}'. Valid: groq, openai, anthropic"
    )


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def _build_prompt(suite: SuiteConfig, test: TestCase) -> str:
    """
    Assemble the full prompt string for a test case.

    If the suite has a system prompt, it is prepended. Variable placeholders
    in the test input are substituted from test.variables.

    Args:
        suite: The suite configuration.
        test:  The individual test case.

    Returns:
        The fully rendered prompt string.
    """
    user_input = test.input
    if test.variables:
        for key, value in test.variables.items():
            user_input = user_input.replace(f"{{{key}}}", str(value))

    system_prompt = suite.get_system_prompt()
    if system_prompt:
        return f"{system_prompt}\n\n{user_input}"
    return user_input


def _prompt_hash(prompt: str) -> str:
    """Return a short SHA-256 hash of *prompt*."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# TestRunner
# ---------------------------------------------------------------------------

class TestRunner:
    """
    Executes a test suite against an LLM, running all enabled evaluators.

    Usage:
        runner = TestRunner()
        results = runner.run_suite(suite, mode="test")
    """

    def __init__(self, storage: Optional[Storage] = None) -> None:
        """
        Args:
            storage: Storage instance (defaults to production SQLite at ~/.promptbench).
        """
        self._storage = storage or Storage()
        self._format_checker = FormatChecker()
        self._length_checker = LengthChecker()
        self._content_checker = ContentChecker()
        self._tone_checker = ToneChecker()
        self._semantic_checker = SemanticChecker()

    def run_suite(
        self,
        suite: SuiteConfig,
        mode: str = "test",
        use_semantic: bool = True,
    ) -> List[TestResult]:
        """
        Run all tests in *suite* and return a list of TestResult instances.

        In 'baseline' mode, outputs are saved as the new baseline without
        running evaluators (always passes). In 'test' mode, outputs are
        compared against stored baselines and evaluators are applied.

        Args:
            suite:        The SuiteConfig to run.
            mode:         'baseline' to record, 'test' to evaluate.
            use_semantic: Whether to run the semantic similarity evaluator.
                          Set False to skip (useful when no baseline exists yet).

        Returns:
            List of TestResult, one per test case.
        """
        api_key = get_api_key(suite.api_provider)
        results: List[TestResult] = []

        for test in suite.tests:
            result = self._run_test(
                suite=suite,
                test=test,
                api_key=api_key,
                mode=mode,
                use_semantic=use_semantic,
            )
            results.append(result)

        return results

    def _run_test(
        self,
        suite: SuiteConfig,
        test: TestCase,
        api_key: str,
        mode: str,
        use_semantic: bool,
    ) -> TestResult:
        """Run a single test case and return its TestResult."""
        prompt = _build_prompt(suite, test)
        start_ms = int(time.monotonic() * 1000)

        try:
            actual = call_llm(prompt, suite.model, suite.api_provider, api_key)
        except Exception as exc:
            end_ms = int(time.monotonic() * 1000)
            return TestResult(
                test_name=test.name,
                passed=False,
                actual_output="",
                expected_output=None,
                duration_ms=end_ms - start_ms,
                failure_reason=f"LLM call failed: {exc}",
            )

        end_ms = int(time.monotonic() * 1000)
        duration = end_ms - start_ms

        if mode == "baseline":
            self._storage.save_baseline(
                suite_name=suite.name,
                test_name=test.name,
                output=actual,
                model=suite.model,
                prompt=prompt,
            )
            return TestResult(
                test_name=test.name,
                passed=True,
                actual_output=actual,
                expected_output=actual,
                duration_ms=duration,
                failure_reason=None,
            )

        # --- Test mode: run evaluators ---
        baseline_record = self._storage.get_baseline(suite.name, test.name)
        baseline_output = baseline_record.output if baseline_record else None

        failures: List[str] = []

        # 1. Format check
        if test.expect.format:
            passed, reason = self._format_checker.check(actual, test.expect.format)
            if not passed and reason:
                failures.append(reason)

        # 2. Length check
        passed, reason = self._length_checker.check(actual, test.expect)
        if not passed and reason:
            failures.append(reason)

        # 3. Content check
        passed, reason = self._content_checker.check(actual, test.expect)
        if not passed and reason:
            failures.append(reason)

        # 4. Tone check
        if test.expect.tone or baseline_output:
            passed, reason = self._tone_checker.check(
                actual,
                test.expect.tone,
                baseline_output if not test.expect.tone else None,
            )
            if not passed and reason:
                failures.append(reason)

        # 5. Semantic similarity check (only when baseline exists)
        if use_semantic and baseline_output:
            passed, reason = self._semantic_checker.check(
                current_output=actual,
                baseline_output=baseline_output,
                threshold=test.expect.semantic_threshold,
            )
            if not passed and reason:
                failures.append(reason)

        if baseline_output is None:
            failures.append(
                "No baseline found for this test. "
                "Run 'promptbench baseline' first."
            )

        overall_passed = len(failures) == 0
        failure_reason = " | ".join(failures) if failures else None

        return TestResult(
            test_name=test.name,
            passed=overall_passed,
            actual_output=actual,
            expected_output=baseline_output,
            duration_ms=duration,
            failure_reason=failure_reason,
        )

    def save_run(
        self,
        suite_name: str,
        results: List[TestResult],
        duration_ms: int,
    ) -> str:
        """
        Persist a completed run and all its results to storage.

        Args:
            suite_name:  Name of the suite that was run.
            results:     List of TestResult instances.
            duration_ms: Total wall-clock duration of the full suite run.

        Returns:
            The generated run_id string.
        """
        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed)
        result_records = [
            ResultRecord(
                run_id="",  # filled by storage
                test_name=r.test_name,
                passed=r.passed,
                actual_output=r.actual_output,
                failure_reason=r.failure_reason,
                duration_ms=r.duration_ms,
            )
            for r in results
        ]
        return self._storage.save_run(
            suite_name=suite_name,
            passed=passed,
            failed=failed,
            duration_ms=duration_ms,
            results=result_records,
        )
