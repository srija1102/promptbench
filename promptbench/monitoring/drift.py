"""
monitoring/drift.py — Detect behavioral drift in production LLM outputs.

Detects three types of drift by comparing recent production logs against
the stored baseline:

  1. Semantic drift   — cosine similarity of sentence embeddings drops
  2. Tone drift       — new hedging/refusal/casual patterns appear
  3. Length drift     — word count deviates significantly from baseline

The drift detector does NOT make live LLM calls — it runs purely on stored data.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import List, Optional

from promptbench.monitoring.ingest import ProductionLog
from promptbench.evaluators.tone_check import analyze_tone
from promptbench.utils.helpers import word_count

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DriftMetric:
    """A single drift measurement for one test."""

    suite_name: str
    test_name: str
    drift_type: str          # 'semantic' | 'tone' | 'length'
    drift_score: float       # Higher = more drift (0.0 = no drift, 1.0 = max)
    threshold: float
    exceeded: bool
    details: str


@dataclass
class DriftReport:
    """Aggregated drift report across all tests in a suite."""

    suite_name: str
    metrics: List[DriftMetric] = field(default_factory=list)

    @property
    def has_drift(self) -> bool:
        return any(m.exceeded for m in self.metrics)

    @property
    def exceeded_metrics(self) -> List[DriftMetric]:
        return [m for m in self.metrics if m.exceeded]


# ---------------------------------------------------------------------------
# Drift detector
# ---------------------------------------------------------------------------

class DriftDetector:
    """
    Analyzes recent production logs and computes drift metrics versus baseline.

    Usage:
        detector = DriftDetector(storage=storage)
        report = detector.analyze_suite("my-suite", window=50)
    """

    # Default thresholds
    SEMANTIC_THRESHOLD = 0.75   # Similarity must stay above this
    LENGTH_DRIFT_PCT = 0.40     # Word count may not drift by more than 40%
    TONE_THRESHOLD = 0.20       # If >20% of samples show new tone violations

    def __init__(self, storage=None) -> None:
        from promptbench.storage import Storage
        self._storage = storage or Storage()

    def analyze_suite(self, suite_name: str, window: int = 50) -> DriftReport:
        """
        Run all drift checks for *suite_name* using the last *window* logs.

        Args:
            suite_name: Suite to analyze.
            window:     Number of recent production logs to consider per test.

        Returns:
            DriftReport with all computed metrics.
        """
        report = DriftReport(suite_name=suite_name)
        baselines = self._storage.list_baselines(suite_name)

        if not baselines:
            logger.warning("No baselines found for suite '%s'. Cannot detect drift.", suite_name)
            return report

        for baseline in baselines:
            logs = self._storage.get_production_logs(
                suite_name, baseline.test_name, limit=window
            )
            if not logs:
                logger.debug(
                    "No production logs for %s/%s — skipping drift check.",
                    suite_name, baseline.test_name,
                )
                continue

            responses = [log.response for log in logs]
            baseline_text = baseline.output

            # 1. Semantic drift
            sem_metric = self._semantic_drift(
                suite_name, baseline.test_name, baseline_text, responses
            )
            if sem_metric:
                report.metrics.append(sem_metric)

            # 2. Tone drift
            tone_metric = self._tone_drift(
                suite_name, baseline.test_name, baseline_text, responses
            )
            if tone_metric:
                report.metrics.append(tone_metric)

            # 3. Length drift
            len_metric = self._length_drift(
                suite_name, baseline.test_name, baseline_text, responses
            )
            if len_metric:
                report.metrics.append(len_metric)

        # Persist metrics
        for metric in report.metrics:
            try:
                self._storage.save_drift_metric(metric)
            except Exception as exc:
                logger.warning("Failed to save drift metric: %s", exc)

        return report

    # ------------------------------------------------------------------
    # Per-dimension drift calculations
    # ------------------------------------------------------------------

    def _semantic_drift(
        self,
        suite_name: str,
        test_name: str,
        baseline_text: str,
        responses: List[str],
    ) -> Optional[DriftMetric]:
        """Compute mean cosine similarity vs. baseline; flag if below threshold."""
        try:
            from promptbench.evaluators.semantic_check import SemanticChecker
            checker = SemanticChecker()
            scores = []
            for resp in responses:
                if resp.strip() and baseline_text.strip():
                    try:
                        sim = checker.similarity(resp, baseline_text)
                        scores.append(sim)
                    except Exception:
                        pass

            if not scores:
                return None

            mean_sim = statistics.mean(scores)
            # Drift score = how far below threshold (clamped to [0, 1])
            drift_score = max(0.0, self.SEMANTIC_THRESHOLD - mean_sim)
            exceeded = mean_sim < self.SEMANTIC_THRESHOLD

            return DriftMetric(
                suite_name=suite_name,
                test_name=test_name,
                drift_type="semantic",
                drift_score=round(drift_score, 4),
                threshold=self.SEMANTIC_THRESHOLD,
                exceeded=exceeded,
                details=(
                    f"Mean semantic similarity: {mean_sim:.3f} "
                    f"(threshold: {self.SEMANTIC_THRESHOLD})"
                ),
            )
        except ImportError:
            logger.debug("sentence-transformers not available; skipping semantic drift.")
            return None
        except Exception as exc:
            logger.warning("Semantic drift check failed: %s", exc)
            return None

    def _tone_drift(
        self,
        suite_name: str,
        test_name: str,
        baseline_text: str,
        responses: List[str],
    ) -> Optional[DriftMetric]:
        """Detect new tone violations (hedging/refusal/casual) vs. baseline."""
        try:
            baseline_profile = analyze_tone(baseline_text)
            violation_count = 0

            for resp in responses:
                current_profile = analyze_tone(resp)
                new_hedging = current_profile.has_hedging and not baseline_profile.has_hedging
                new_refusal = current_profile.has_refusals and not baseline_profile.has_refusals
                new_casual = current_profile.has_casual and not baseline_profile.has_casual
                if new_hedging or new_refusal or new_casual:
                    violation_count += 1

            violation_rate = violation_count / len(responses)
            exceeded = violation_rate > self.TONE_THRESHOLD

            return DriftMetric(
                suite_name=suite_name,
                test_name=test_name,
                drift_type="tone",
                drift_score=round(violation_rate, 4),
                threshold=self.TONE_THRESHOLD,
                exceeded=exceeded,
                details=(
                    f"{violation_count}/{len(responses)} samples show new tone violations "
                    f"({violation_rate:.1%})"
                ),
            )
        except Exception as exc:
            logger.warning("Tone drift check failed: %s", exc)
            return None

    def _length_drift(
        self,
        suite_name: str,
        test_name: str,
        baseline_text: str,
        responses: List[str],
    ) -> Optional[DriftMetric]:
        """Detect significant word-count drift vs. baseline."""
        try:
            baseline_words = word_count(baseline_text)
            if baseline_words == 0:
                return None

            counts = [word_count(r) for r in responses if r.strip()]
            if not counts:
                return None

            mean_count = statistics.mean(counts)
            pct_change = abs(mean_count - baseline_words) / baseline_words
            exceeded = pct_change > self.LENGTH_DRIFT_PCT

            return DriftMetric(
                suite_name=suite_name,
                test_name=test_name,
                drift_type="length",
                drift_score=round(pct_change, 4),
                threshold=self.LENGTH_DRIFT_PCT,
                exceeded=exceeded,
                details=(
                    f"Baseline: {baseline_words} words; "
                    f"Recent mean: {mean_count:.0f} words ({pct_change:.1%} change)"
                ),
            )
        except Exception as exc:
            logger.warning("Length drift check failed: %s", exc)
            return None
