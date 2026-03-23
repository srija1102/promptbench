"""
test_drift.py — Unit tests for drift detection and alerting.

No LLM API calls are made — tests operate purely on stored data.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from promptbench.storage import Storage
from promptbench.monitoring.ingest import LogIngester, ProductionLog
from promptbench.monitoring.drift import DriftDetector, DriftReport, DriftMetric
from promptbench.monitoring.alerts import AlertManager, AlertEvent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_storage() -> Storage:
    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    tmp.close()
    return Storage(db_path=Path(tmp.name))


def _seed_baseline(storage: Storage, suite: str, test: str, output: str) -> None:
    storage.save_baseline(suite, test, output, "test-model", "prompt")


def _seed_logs(
    storage: Storage,
    suite: str,
    test: str,
    responses: list,
    model: str = "test-model",
) -> None:
    ingester = LogIngester(storage=storage)
    for resp in responses:
        ingester.ingest(
            suite_name=suite,
            test_name=test,
            prompt="test prompt",
            response=resp,
            model=model,
        )


# ===========================================================================
# LogIngester
# ===========================================================================

class TestLogIngester:

    def test_ingest_stores_log(self):
        storage = _make_storage()
        ingester = LogIngester(storage=storage)
        ingester.ingest(
            suite_name="suite1",
            test_name="test1",
            prompt="Hello?",
            response="Hi there!",
            model="test-model",
        )
        logs = storage.get_production_logs("suite1", "test1", limit=10)
        assert len(logs) == 1
        assert logs[0].response == "Hi there!"

    def test_ingest_multiple_logs(self):
        storage = _make_storage()
        ingester = LogIngester(storage=storage)
        for i in range(5):
            ingester.ingest("suite", "test", f"prompt {i}", f"response {i}", "model")
        logs = storage.get_production_logs("suite", "test", limit=10)
        assert len(logs) == 5

    def test_ingest_batch(self):
        storage = _make_storage()
        ingester = LogIngester(storage=storage)
        logs = [
            ProductionLog(
                suite_name="s", test_name="t",
                prompt=f"p{i}", response=f"r{i}",
                model="m", timestamp=1000.0 + i,
            )
            for i in range(3)
        ]
        count = ingester.ingest_batch(logs)
        assert count == 3

    def test_ingest_with_metadata(self):
        storage = _make_storage()
        ingester = LogIngester(storage=storage)
        ingester.ingest(
            suite_name="suite", test_name="test",
            prompt="p", response="r", model="m",
            metadata={"user_id": "123", "env": "prod"},
        )
        logs = storage.get_production_logs("suite", "test")
        assert logs[0].metadata is not None
        assert "user_id" in logs[0].metadata

    def test_get_recent_logs_respects_limit(self):
        storage = _make_storage()
        ingester = LogIngester(storage=storage)
        for i in range(20):
            ingester.ingest("s", "t", f"p{i}", f"r{i}", "m")
        logs = ingester.get_recent_logs("s", "t", limit=5)
        assert len(logs) == 5


# ===========================================================================
# DriftDetector — length drift
# ===========================================================================

class TestLengthDrift:

    def test_no_length_drift_when_similar_to_baseline(self):
        storage = _make_storage()
        baseline_text = "The capital of France is Paris and it is a beautiful city."
        _seed_baseline(storage, "s", "t", baseline_text)
        # Seed similar-length responses
        _seed_logs(storage, "s", "t", [baseline_text] * 10)

        detector = DriftDetector(storage=storage)
        report = detector.analyze_suite("s", window=50)

        len_metrics = [m for m in report.metrics if m.drift_type == "length"]
        assert all(not m.exceeded for m in len_metrics)

    def test_length_drift_detected_when_responses_too_short(self):
        storage = _make_storage()
        baseline_text = " ".join(["word"] * 100)  # 100-word baseline
        _seed_baseline(storage, "s", "t", baseline_text)
        # Seed very short responses (90%+ shorter)
        _seed_logs(storage, "s", "t", ["Hi."] * 20)

        detector = DriftDetector(storage=storage)
        report = detector.analyze_suite("s", window=50)

        len_metrics = [m for m in report.metrics if m.drift_type == "length"]
        assert any(m.exceeded for m in len_metrics)

    def test_no_drift_with_no_logs(self):
        storage = _make_storage()
        _seed_baseline(storage, "s", "t", "The capital of France is Paris.")
        # No production logs seeded

        detector = DriftDetector(storage=storage)
        report = detector.analyze_suite("s", window=50)

        # Should have no metrics (no logs to compare against)
        assert len(report.metrics) == 0

    def test_no_drift_with_no_baselines(self):
        storage = _make_storage()
        # No baseline, no logs

        detector = DriftDetector(storage=storage)
        report = detector.analyze_suite("s", window=50)

        assert len(report.metrics) == 0
        assert report.has_drift is False


# ===========================================================================
# DriftDetector — tone drift
# ===========================================================================

class TestToneDrift:

    def test_tone_drift_detected_new_refusals(self):
        storage = _make_storage()
        baseline_text = "The capital of France is Paris."
        _seed_baseline(storage, "s", "t", baseline_text)
        # Production logs now contain refusals
        _seed_logs(
            storage, "s", "t",
            ["As an AI, I cannot answer this question."] * 10
        )

        detector = DriftDetector(storage=storage)
        report = detector.analyze_suite("s", window=50)

        tone_metrics = [m for m in report.metrics if m.drift_type == "tone"]
        assert any(m.exceeded for m in tone_metrics)

    def test_no_tone_drift_when_same_pattern(self):
        storage = _make_storage()
        hedgy = "I think maybe the answer is Paris."
        _seed_baseline(storage, "s", "t", hedgy)
        # Production logs also hedgy — same as baseline
        _seed_logs(storage, "s", "t", [hedgy] * 10)

        detector = DriftDetector(storage=storage)
        report = detector.analyze_suite("s", window=50)

        tone_metrics = [m for m in report.metrics if m.drift_type == "tone"]
        assert all(not m.exceeded for m in tone_metrics)


# ===========================================================================
# DriftReport
# ===========================================================================

class TestDriftReport:

    def test_has_drift_false_when_no_exceeded(self):
        report = DriftReport(suite_name="s", metrics=[
            DriftMetric("s", "t", "length", 0.1, 0.4, False, "OK"),
        ])
        assert report.has_drift is False

    def test_has_drift_true_when_any_exceeded(self):
        report = DriftReport(suite_name="s", metrics=[
            DriftMetric("s", "t", "length", 0.5, 0.4, True, "Exceeded"),
        ])
        assert report.has_drift is True

    def test_exceeded_metrics_filters_correctly(self):
        report = DriftReport(suite_name="s", metrics=[
            DriftMetric("s", "t1", "length", 0.1, 0.4, False, "OK"),
            DriftMetric("s", "t2", "tone", 0.5, 0.2, True, "Exceeded"),
        ])
        exceeded = report.exceeded_metrics
        assert len(exceeded) == 1
        assert exceeded[0].test_name == "t2"


# ===========================================================================
# AlertManager
# ===========================================================================

class TestAlertManager:

    def test_no_alerts_when_no_drift(self):
        report = DriftReport(suite_name="s", metrics=[
            DriftMetric("s", "t", "length", 0.1, 0.4, False, "OK"),
        ])
        manager = AlertManager()
        events = manager.evaluate(report)
        assert len(events) == 0

    def test_alert_triggered_when_exceeded(self):
        report = DriftReport(suite_name="s", metrics=[
            DriftMetric("s", "t", "tone", 0.5, 0.2, True, "Refusal rate high"),
        ])
        manager = AlertManager()
        events = manager.evaluate(report)
        assert len(events) == 1
        assert events[0].drift_type == "tone"
        assert "DRIFT" in events[0].message or "tone" in events[0].message.lower()

    def test_multiple_alerts_fired(self):
        report = DriftReport(suite_name="s", metrics=[
            DriftMetric("s", "t1", "length", 0.6, 0.4, True, "Too short"),
            DriftMetric("s", "t2", "tone", 0.4, 0.2, True, "New refusals"),
            DriftMetric("s", "t3", "semantic", 0.05, 0.75, False, "OK"),
        ])
        manager = AlertManager()
        events = manager.evaluate(report)
        assert len(events) == 2

    def test_webhook_called_when_configured(self):
        report = DriftReport(suite_name="s", metrics=[
            DriftMetric("s", "t", "length", 0.6, 0.4, True, "Exceeded"),
        ])
        manager = AlertManager(webhook_url="http://example.com/webhook")
        with patch("requests.post") as mock_post:
            mock_post.return_value = MagicMock(ok=True)
            manager.evaluate(report)
            assert mock_post.called

    def test_webhook_not_called_when_not_configured(self):
        report = DriftReport(suite_name="s", metrics=[
            DriftMetric("s", "t", "length", 0.6, 0.4, True, "Exceeded"),
        ])
        manager = AlertManager(webhook_url=None, slack_webhook_url=None)
        with patch("requests.post") as mock_post:
            manager.evaluate(report)
            assert not mock_post.called
