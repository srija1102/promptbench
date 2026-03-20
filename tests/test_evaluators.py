"""
test_evaluators.py — Unit tests for all evaluator modules.

Tests use real string inputs — no mocking required.
At least 3 tests per evaluator (format, length, content, tone).
"""

from __future__ import annotations

import pytest

from promptbench.config import ExpectConfig
from promptbench.evaluators.format_check import FormatChecker
from promptbench.evaluators.length_check import LengthChecker
from promptbench.evaluators.content_check import ContentChecker
from promptbench.evaluators.tone_check import ToneChecker, analyze_tone


# ===========================================================================
# FormatChecker
# ===========================================================================

class TestFormatChecker:
    """Tests for format_check.FormatChecker."""

    def setup_method(self):
        self.checker = FormatChecker()

    # --- bullet_points ---

    def test_bullet_points_passes_with_dashes(self):
        output = "- First item\n- Second item\n- Third item"
        passed, reason = self.checker.check(output, "bullet_points")
        assert passed is True
        assert reason is None

    def test_bullet_points_passes_with_asterisks(self):
        output = "* Red\n* Blue\n* Green"
        passed, reason = self.checker.check(output, "bullet_points")
        assert passed is True

    def test_bullet_points_passes_with_bullet_character(self):
        output = "• Apple\n• Banana\n• Cherry"
        passed, reason = self.checker.check(output, "bullet_points")
        assert passed is True

    def test_bullet_points_fails_on_plain_text(self):
        output = "The sky is blue. The grass is green."
        passed, reason = self.checker.check(output, "bullet_points")
        assert passed is False
        assert reason is not None
        assert "bullet" in reason.lower()

    # --- json ---

    def test_json_passes_with_valid_object(self):
        output = '{"sentiment": "positive", "confidence": 0.92}'
        passed, reason = self.checker.check(output, "json")
        assert passed is True

    def test_json_passes_with_array(self):
        output = '[{"name": "Alice"}, {"name": "Bob"}]'
        passed, reason = self.checker.check(output, "json")
        assert passed is True

    def test_json_passes_with_fenced_code_block(self):
        output = "Here is the result:\n```json\n{\"key\": \"value\"}\n```"
        passed, reason = self.checker.check(output, "json")
        assert passed is True

    def test_json_fails_on_invalid_json(self):
        output = "This is not JSON at all."
        passed, reason = self.checker.check(output, "json")
        assert passed is False
        assert "JSON" in reason

    def test_json_fails_on_malformed_json(self):
        output = '{"name": "Alice", "age":}'
        passed, reason = self.checker.check(output, "json")
        assert passed is False

    # --- numbered_list ---

    def test_numbered_list_passes(self):
        output = "1. Buy groceries\n2. Cook dinner\n3. Wash dishes"
        passed, reason = self.checker.check(output, "numbered_list")
        assert passed is True

    def test_numbered_list_fails_on_bullets(self):
        output = "- item one\n- item two"
        passed, reason = self.checker.check(output, "numbered_list")
        assert passed is False

    def test_numbered_list_fails_on_plain_text(self):
        output = "First do this. Then do that."
        passed, reason = self.checker.check(output, "numbered_list")
        assert passed is False

    # --- markdown ---

    def test_markdown_passes_with_heading(self):
        output = "# Introduction\nThis is the intro."
        passed, reason = self.checker.check(output, "markdown")
        assert passed is True

    def test_markdown_passes_with_bold(self):
        output = "This is **very important** text."
        passed, reason = self.checker.check(output, "markdown")
        assert passed is True

    def test_markdown_passes_with_code_block(self):
        output = "Run this:\n```python\nprint('hello')\n```"
        passed, reason = self.checker.check(output, "markdown")
        assert passed is True

    def test_markdown_fails_on_plain_text(self):
        output = "No formatting here at all. Just plain words."
        passed, reason = self.checker.check(output, "markdown")
        assert passed is False

    # --- plain_text ---

    def test_plain_text_passes(self):
        output = "The answer is four. No special characters here."
        passed, reason = self.checker.check(output, "plain_text")
        assert passed is True

    def test_plain_text_fails_with_markdown(self):
        output = "# Heading\nSome **bold** text."
        passed, reason = self.checker.check(output, "plain_text")
        assert passed is False

    # --- no format specified ---

    def test_none_format_always_passes(self):
        passed, reason = self.checker.check("anything", None)
        assert passed is True
        assert reason is None

    # --- unknown format ---

    def test_unknown_format_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown format"):
            self.checker.check("output", "csv")


# ===========================================================================
# LengthChecker
# ===========================================================================

class TestLengthChecker:
    """Tests for length_check.LengthChecker."""

    def setup_method(self):
        self.checker = LengthChecker()

    def _expect(self, **kwargs) -> ExpectConfig:
        return ExpectConfig(**kwargs)

    def test_word_count_within_limits_passes(self):
        output = "The quick brown fox jumps over the lazy dog."
        passed, reason = self.checker.check(output, self._expect(min_words=5, max_words=20))
        assert passed is True

    def test_word_count_exceeds_max_fails(self):
        output = " ".join(["word"] * 30)
        passed, reason = self.checker.check(output, self._expect(max_words=10))
        assert passed is False
        assert "max_words" in reason

    def test_word_count_below_min_fails(self):
        output = "Short."
        passed, reason = self.checker.check(output, self._expect(min_words=20))
        assert passed is False
        assert "min_words" in reason

    def test_bullet_count_passes(self):
        output = "- Item 1\n- Item 2\n- Item 3"
        passed, reason = self.checker.check(output, self._expect(min_bullets=2, max_bullets=5))
        assert passed is True

    def test_bullet_count_exceeds_max_fails(self):
        output = "- A\n- B\n- C\n- D\n- E\n- F"
        passed, reason = self.checker.check(output, self._expect(max_bullets=3))
        assert passed is False
        assert "max_bullets" in reason

    def test_bullet_count_below_min_fails(self):
        output = "- Only one bullet"
        passed, reason = self.checker.check(output, self._expect(min_bullets=3))
        assert passed is False
        assert "min_bullets" in reason

    def test_sentence_count_passes(self):
        output = "First sentence. Second sentence. Third sentence."
        passed, reason = self.checker.check(output, self._expect(max_sentences=5))
        assert passed is True

    def test_sentence_count_exceeds_max_fails(self):
        output = "One. Two. Three. Four. Five. Six. Seven."
        passed, reason = self.checker.check(output, self._expect(max_sentences=3))
        assert passed is False
        assert "max_sentences" in reason

    def test_multiple_constraints_combined_failure(self):
        output = "hi"
        passed, reason = self.checker.check(
            output, self._expect(min_words=20, min_bullets=3)
        )
        assert passed is False
        # Both failures should appear in the reason
        assert "min_words" in reason
        assert "min_bullets" in reason

    def test_no_constraints_always_passes(self):
        output = "Any output at all."
        passed, reason = self.checker.check(output, ExpectConfig())
        assert passed is True

    def test_word_count_utility(self):
        assert LengthChecker.word_count("hello world") == 2

    def test_bullet_count_utility(self):
        assert LengthChecker.bullet_count("- a\n- b\n- c") == 3

    def test_sentence_count_utility(self):
        count = LengthChecker.sentence_count("Hello. World. Foo.")
        assert count == 3


# ===========================================================================
# ContentChecker
# ===========================================================================

class TestContentChecker:
    """Tests for content_check.ContentChecker."""

    def setup_method(self):
        self.checker = ContentChecker()

    def _expect(self, **kwargs) -> ExpectConfig:
        return ExpectConfig(**kwargs)

    def test_required_phrase_present_passes(self):
        output = "Machine learning is transforming healthcare."
        passed, reason = self.checker.check(
            output, self._expect(required_phrases=["machine learning"])
        )
        assert passed is True

    def test_required_phrase_case_insensitive(self):
        output = "MACHINE LEARNING is awesome."
        passed, reason = self.checker.check(
            output, self._expect(required_phrases=["machine learning"])
        )
        assert passed is True

    def test_required_phrase_missing_fails(self):
        output = "This output says nothing relevant."
        passed, reason = self.checker.check(
            output, self._expect(required_phrases=["quantum computing"])
        )
        assert passed is False
        assert "quantum computing" in reason

    def test_forbidden_phrase_absent_passes(self):
        output = "The answer is definitely 42."
        passed, reason = self.checker.check(
            output, self._expect(forbidden_phrases=["I cannot"])
        )
        assert passed is True

    def test_forbidden_phrase_present_fails(self):
        output = "I cannot answer that question."
        passed, reason = self.checker.check(
            output, self._expect(forbidden_phrases=["I cannot"])
        )
        assert passed is False
        assert "I cannot" in reason

    def test_forbidden_phrase_case_insensitive(self):
        output = "AS AN AI, I must inform you..."
        passed, reason = self.checker.check(
            output, self._expect(forbidden_phrases=["as an ai"])
        )
        assert passed is False

    def test_required_pattern_matches_passes(self):
        output = '{"sentiment": "positive", "confidence": 0.9}'
        passed, reason = self.checker.check(
            output, self._expect(required_patterns=[r'"sentiment"\s*:\s*"positive"'])
        )
        assert passed is True

    def test_required_pattern_no_match_fails(self):
        output = "The sentiment is positive."
        passed, reason = self.checker.check(
            output, self._expect(required_patterns=[r'"sentiment"\s*:\s*"positive"'])
        )
        assert passed is False

    def test_forbidden_pattern_no_match_passes(self):
        output = "Here is the information you requested."
        passed, reason = self.checker.check(
            output, self._expect(forbidden_patterns=[r"I (cannot|can't)"])
        )
        assert passed is True

    def test_forbidden_pattern_matches_fails(self):
        output = "I cannot provide that information."
        passed, reason = self.checker.check(
            output, self._expect(forbidden_patterns=[r"I (cannot|can't)"])
        )
        assert passed is False
        assert "I cannot" in reason

    def test_invalid_regex_fails_gracefully(self):
        output = "Some output"
        passed, reason = self.checker.check(
            output, self._expect(required_patterns=["[invalid regex"])
        )
        assert passed is False
        assert "Invalid" in reason

    def test_multiple_requirements_combined(self):
        output = "The capital of France is Paris."
        passed, reason = self.checker.check(
            output,
            self._expect(
                required_phrases=["Paris"],
                forbidden_phrases=["London"],
                required_patterns=[r"\bFrance\b"],
            ),
        )
        assert passed is True

    def test_no_constraints_always_passes(self):
        passed, reason = self.checker.check("anything", ExpectConfig())
        assert passed is True


# ===========================================================================
# ToneChecker
# ===========================================================================

class TestToneChecker:
    """Tests for tone_check.ToneChecker."""

    def setup_method(self):
        self.checker = ToneChecker()

    def test_professional_tone_clean_passes(self):
        output = "The analysis indicates a 15% improvement in throughput."
        passed, reason = self.checker.check(output, "professional")
        assert passed is True

    def test_professional_tone_with_hedging_fails(self):
        output = "I think this might possibly be the answer."
        passed, reason = self.checker.check(output, "professional")
        assert passed is False
        assert "hedging" in reason.lower()

    def test_professional_tone_with_refusal_fails(self):
        output = "As an AI, I cannot provide financial advice."
        passed, reason = self.checker.check(output, "professional")
        assert passed is False
        assert "refusal" in reason.lower()

    def test_professional_tone_with_casual_fails(self):
        output = "Yeah, gonna say the answer is probably 42."
        passed, reason = self.checker.check(output, "professional")
        assert passed is False

    def test_casual_tone_allows_casual_language(self):
        output = "Hey! That's a cool question. Yeah, you're gonna love this answer."
        passed, reason = self.checker.check(output, "casual")
        assert passed is True

    def test_neutral_tone_clean_passes(self):
        output = "The temperature today is 22 degrees Celsius."
        passed, reason = self.checker.check(output, "neutral")
        assert passed is True

    def test_baseline_drift_no_new_violations_passes(self):
        baseline = "I think the answer might be 42."
        current = "I think the answer might be 42 as well."
        # Both have hedging, so no NEW violations
        passed, reason = self.checker.check(current, None, baseline_output=baseline)
        assert passed is True

    def test_baseline_drift_new_hedging_fails(self):
        baseline = "The answer is definitively 42."
        current = "I think maybe the answer could possibly be 42."
        passed, reason = self.checker.check(current, None, baseline_output=baseline)
        assert passed is False
        assert "hedging" in reason.lower()

    def test_baseline_drift_new_refusal_fails(self):
        baseline = "Here is the information about X."
        current = "As an AI, I cannot provide information about X."
        passed, reason = self.checker.check(current, None, baseline_output=baseline)
        assert passed is False
        assert "refusal" in reason.lower()

    def test_new_casual_language_drift_fails(self):
        baseline = "The recommendation is to proceed carefully."
        current = "Yeah, gonna be honest — you should probably just go for it."
        passed, reason = self.checker.check(current, None, baseline_output=baseline)
        assert passed is False
        assert "casual" in reason.lower()

    def test_no_tone_no_baseline_passes(self):
        passed, reason = self.checker.check("anything", None, None)
        assert passed is True

    def test_analyze_utility_detects_hedging(self):
        profile = ToneChecker.analyze("I think maybe this is correct.")
        assert profile.has_hedging is True

    def test_analyze_utility_detects_refusal(self):
        profile = ToneChecker.analyze("As an AI, I cannot do that.")
        assert profile.has_refusals is True

    def test_analyze_utility_detects_casual(self):
        profile = ToneChecker.analyze("Yeah, that's gonna be awesome!")
        assert profile.has_casual is True

    def test_analyze_utility_clean_text(self):
        profile = ToneChecker.analyze("The quarterly results exceeded expectations.")
        assert profile.has_hedging is False
        assert profile.has_refusals is False
        assert profile.has_casual is False
