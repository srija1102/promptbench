"""
test_runner.py — Integration tests for the TestRunner with mocked LLM responses.

All LLM API calls are intercepted using unittest.mock so no API key is needed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from promptbench.config import load_suite, SuiteConfig, TestCase, ExpectConfig
from promptbench.runner import TestRunner, call_llm
from promptbench.storage import Storage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURE_SUITE = Path(__file__).parent / "fixtures" / "sample_suite.yaml"


def _make_storage() -> Storage:
    """Create an in-memory / temp-file Storage for tests."""
    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    tmp.close()
    return Storage(db_path=Path(tmp.name))


def _make_simple_suite(name: str = "test-suite") -> SuiteConfig:
    """Create a minimal SuiteConfig without loading from disk."""
    return SuiteConfig(
        name=name,
        model="llama3-8b-8192",
        api_provider="groq",
        tests=[
            TestCase(
                name="test-one",
                input="What is 2 + 2?",
                expect=ExpectConfig(
                    required_phrases=["4"],
                    max_words=50,
                ),
            ),
            TestCase(
                name="test-two",
                input="List 3 colors as bullets.",
                expect=ExpectConfig(
                    format="bullet_points",
                    min_bullets=2,
                ),
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Test: successful baseline + test run
# ---------------------------------------------------------------------------

class TestSuccessfulRun:
    """Tests for happy-path execution where LLM returns good outputs."""

    @patch("promptbench.runner.get_api_key", return_value="test-key")
    @patch("promptbench.runner.call_llm")
    def test_baseline_saves_and_returns_passing_results(self, mock_call_llm, mock_get_key):
        """Baseline mode should always return passed=True and save to storage."""
        mock_call_llm.return_value = "The answer is 4."
        storage = _make_storage()
        runner = TestRunner(storage=storage)
        suite = _make_simple_suite()

        results = runner.run_suite(suite, mode="baseline")

        assert len(results) == 2
        assert all(r.passed for r in results), "Baseline mode must always pass"
        assert all(r.actual_output == "The answer is 4." for r in results)

        # Verify baselines were persisted
        for test in suite.tests:
            record = storage.get_baseline(suite.name, test.name)
            assert record is not None
            assert record.output == "The answer is 4."

    @patch("promptbench.runner.get_api_key", return_value="test-key")
    @patch("promptbench.runner.call_llm")
    def test_test_mode_passes_when_output_matches_expect(self, mock_call_llm, mock_get_key):
        """Test mode should pass when output satisfies all constraints."""
        storage = _make_storage()
        runner = TestRunner(storage=storage)
        suite = SuiteConfig(
            name="passing-suite",
            model="llama3-8b-8192",
            api_provider="groq",
            tests=[
                TestCase(
                    name="simple-math",
                    input="What is 2 + 2?",
                    expect=ExpectConfig(
                        required_phrases=["4"],
                        max_words=30,
                        tone="professional",
                    ),
                )
            ],
        )

        good_output = "The result is 4."

        # First, save a baseline
        mock_call_llm.return_value = good_output
        runner.run_suite(suite, mode="baseline")

        # Now test against it — same output
        mock_call_llm.return_value = good_output
        results = runner.run_suite(suite, mode="test", use_semantic=False)

        assert len(results) == 1
        assert results[0].passed is True
        assert results[0].failure_reason is None

    @patch("promptbench.runner.get_api_key", return_value="test-key")
    @patch("promptbench.runner.call_llm")
    def test_run_is_persisted_to_storage(self, mock_call_llm, mock_get_key):
        """save_run() should store a RunRecord retrievable via get_run_history."""
        mock_call_llm.return_value = "The answer is 4."
        storage = _make_storage()
        runner = TestRunner(storage=storage)
        suite = _make_simple_suite("history-suite")

        runner.run_suite(suite, mode="baseline")
        results = runner.run_suite(suite, mode="test", use_semantic=False)
        run_id = runner.save_run(suite.name, results, duration_ms=500)

        history = storage.get_run_history(suite.name)
        assert len(history) >= 1
        assert history[0].suite_name == "history-suite"


# ---------------------------------------------------------------------------
# Test: failing run
# ---------------------------------------------------------------------------

class TestFailingRun:
    """Tests for runs where LLM output violates expectations."""

    @patch("promptbench.runner.get_api_key", return_value="test-key")
    @patch("promptbench.runner.call_llm")
    def test_test_mode_fails_on_missing_required_phrase(self, mock_call_llm, mock_get_key):
        """Test should fail when a required phrase is absent from output."""
        storage = _make_storage()
        runner = TestRunner(storage=storage)
        suite = SuiteConfig(
            name="failing-suite",
            model="llama3-8b-8192",
            api_provider="groq",
            tests=[
                TestCase(
                    name="must-mention-paris",
                    input="What is the capital of France?",
                    expect=ExpectConfig(
                        required_phrases=["Paris"],
                    ),
                )
            ],
        )

        # Baseline: correct answer
        mock_call_llm.return_value = "The capital of France is Paris."
        runner.run_suite(suite, mode="baseline")

        # Test: wrong answer — regression!
        mock_call_llm.return_value = "The capital of France is Lyon."
        results = runner.run_suite(suite, mode="test", use_semantic=False)

        assert len(results) == 1
        assert results[0].passed is False
        assert results[0].failure_reason is not None
        assert "Paris" in results[0].failure_reason

    @patch("promptbench.runner.get_api_key", return_value="test-key")
    @patch("promptbench.runner.call_llm")
    def test_test_mode_fails_on_forbidden_phrase(self, mock_call_llm, mock_get_key):
        """Test should fail when a forbidden phrase appears in output."""
        storage = _make_storage()
        runner = TestRunner(storage=storage)
        suite = SuiteConfig(
            name="forbidden-suite",
            model="llama3-8b-8192",
            api_provider="groq",
            tests=[
                TestCase(
                    name="no-refusals",
                    input="Explain photosynthesis.",
                    expect=ExpectConfig(
                        forbidden_phrases=["As an AI"],
                    ),
                )
            ],
        )

        # Baseline: good output
        mock_call_llm.return_value = "Photosynthesis converts sunlight into glucose."
        runner.run_suite(suite, mode="baseline")

        # Test: regression — model now refuses
        mock_call_llm.return_value = "As an AI, I cannot explain scientific processes."
        results = runner.run_suite(suite, mode="test", use_semantic=False)

        assert results[0].passed is False
        assert "As an AI" in results[0].failure_reason

    @patch("promptbench.runner.get_api_key", return_value="test-key")
    @patch("promptbench.runner.call_llm")
    def test_test_mode_fails_on_format_violation(self, mock_call_llm, mock_get_key):
        """Test should fail when output doesn't match expected format."""
        storage = _make_storage()
        runner = TestRunner(storage=storage)
        suite = SuiteConfig(
            name="format-suite",
            model="llama3-8b-8192",
            api_provider="groq",
            tests=[
                TestCase(
                    name="must-be-bullets",
                    input="List three fruits.",
                    expect=ExpectConfig(format="bullet_points"),
                )
            ],
        )

        # Baseline: correct bullet output
        mock_call_llm.return_value = "- Apple\n- Banana\n- Cherry"
        runner.run_suite(suite, mode="baseline")

        # Test: regression — now plain text
        mock_call_llm.return_value = "Apple, Banana, and Cherry are three fruits."
        results = runner.run_suite(suite, mode="test", use_semantic=False)

        assert results[0].passed is False
        assert "bullet" in results[0].failure_reason.lower()

    @patch("promptbench.runner.get_api_key", return_value="test-key")
    @patch("promptbench.runner.call_llm")
    def test_test_mode_fails_on_word_count_violation(self, mock_call_llm, mock_get_key):
        """Test should fail when output exceeds max_words constraint."""
        storage = _make_storage()
        runner = TestRunner(storage=storage)
        suite = SuiteConfig(
            name="length-suite",
            model="llama3-8b-8192",
            api_provider="groq",
            tests=[
                TestCase(
                    name="brief-answer",
                    input="What is 1+1?",
                    expect=ExpectConfig(max_words=5),
                )
            ],
        )

        mock_call_llm.return_value = "2"
        runner.run_suite(suite, mode="baseline")

        # Regression: extremely verbose output
        mock_call_llm.return_value = " ".join(["word"] * 50)
        results = runner.run_suite(suite, mode="test", use_semantic=False)

        assert results[0].passed is False
        assert "max_words" in results[0].failure_reason


# ---------------------------------------------------------------------------
# Test: missing baseline
# ---------------------------------------------------------------------------

class TestMissingBaseline:
    """Tests for behavior when no baseline has been saved."""

    @patch("promptbench.runner.get_api_key", return_value="test-key")
    @patch("promptbench.runner.call_llm")
    def test_test_without_baseline_fails_with_clear_message(self, mock_call_llm, mock_get_key):
        """Running test mode before baseline should fail with a helpful message."""
        mock_call_llm.return_value = "Some output."
        storage = _make_storage()
        runner = TestRunner(storage=storage)
        suite = SuiteConfig(
            name="no-baseline-suite",
            model="llama3-8b-8192",
            api_provider="groq",
            tests=[
                TestCase(
                    name="orphaned-test",
                    input="Hello?",
                    expect=ExpectConfig(),
                )
            ],
        )

        results = runner.run_suite(suite, mode="test", use_semantic=False)

        assert len(results) == 1
        assert results[0].passed is False
        assert "baseline" in results[0].failure_reason.lower()

    @patch("promptbench.runner.get_api_key", return_value="test-key")
    @patch("promptbench.runner.call_llm")
    def test_baseline_then_test_without_baseline_test_fails(self, mock_call_llm, mock_get_key):
        """If baseline exists for one test but not another, the missing one fails."""
        storage = _make_storage()
        runner = TestRunner(storage=storage)
        suite = SuiteConfig(
            name="partial-baseline",
            model="llama3-8b-8192",
            api_provider="groq",
            tests=[
                TestCase(name="has-baseline", input="Q1?", expect=ExpectConfig()),
                TestCase(name="no-baseline", input="Q2?", expect=ExpectConfig()),
            ],
        )

        # Save baseline only for the first test
        mock_call_llm.return_value = "Answer."
        storage.save_baseline("partial-baseline", "has-baseline", "Answer.", "llama3-8b-8192", "Q1?")

        results = runner.run_suite(suite, mode="test", use_semantic=False)

        by_name = {r.test_name: r for r in results}
        assert by_name["no-baseline"].passed is False
        assert "baseline" in by_name["no-baseline"].failure_reason.lower()

    @patch("promptbench.runner.get_api_key", return_value="test-key")
    @patch("promptbench.runner.call_llm")
    def test_llm_call_failure_returns_failing_result(self, mock_call_llm, mock_get_key):
        """When the LLM API call throws, the test should fail gracefully."""
        mock_call_llm.side_effect = ConnectionError("Network unreachable")
        storage = _make_storage()
        runner = TestRunner(storage=storage)
        suite = SuiteConfig(
            name="api-error-suite",
            model="llama3-8b-8192",
            api_provider="groq",
            tests=[
                TestCase(name="will-fail", input="Test?", expect=ExpectConfig())
            ],
        )

        results = runner.run_suite(suite, mode="test", use_semantic=False)

        assert results[0].passed is False
        assert "LLM call failed" in results[0].failure_reason


# ---------------------------------------------------------------------------
# Test: fixture file loading
# ---------------------------------------------------------------------------

class TestFixtureLoading:
    """Test that the sample fixture YAML loads correctly."""

    def test_sample_suite_loads(self):
        """The fixture YAML should load without errors."""
        suite = load_suite(FIXTURE_SUITE)
        assert suite.name == "sample-test-suite"
        assert len(suite.tests) == 3

    def test_sample_suite_test_names(self):
        """Each test in the fixture has the expected name."""
        suite = load_suite(FIXTURE_SUITE)
        names = [t.name for t in suite.tests]
        assert "basic-response" in names
        assert "format-test" in names
        assert "forbidden-content" in names

    def test_sample_suite_expect_config(self):
        """The fixture test expectations are parsed correctly."""
        suite = load_suite(FIXTURE_SUITE)
        basic = next(t for t in suite.tests if t.name == "basic-response")
        assert "4" in basic.expect.required_phrases
        assert basic.expect.max_words == 50
