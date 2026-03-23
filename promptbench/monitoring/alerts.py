"""
monitoring/alerts.py — Alert triggers when drift exceeds configured thresholds.

Supports:
  - Console (stderr) alerts — always on
  - Webhook alerts — POST JSON payload to a URL (set PROMPTBENCH_WEBHOOK_URL)
  - Slack alerts — via incoming webhook (set PROMPTBENCH_SLACK_WEBHOOK_URL)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import List, Optional

from promptbench.monitoring.drift import DriftMetric, DriftReport

logger = logging.getLogger(__name__)


@dataclass
class AlertEvent:
    """Represents a triggered alert for a single drift metric."""

    suite_name: str
    test_name: str
    drift_type: str
    drift_score: float
    threshold: float
    details: str
    message: str


class AlertManager:
    """
    Evaluates a DriftReport and dispatches alerts for exceeded thresholds.

    Usage:
        manager = AlertManager()
        alerts = manager.evaluate(report)
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        slack_webhook_url: Optional[str] = None,
    ) -> None:
        """
        Args:
            webhook_url:       HTTP endpoint to POST JSON alerts to (optional).
            slack_webhook_url: Slack incoming webhook URL (optional).
        """
        self._webhook_url = (
            webhook_url
            or os.environ.get("PROMPTBENCH_WEBHOOK_URL")
        )
        self._slack_url = (
            slack_webhook_url
            or os.environ.get("PROMPTBENCH_SLACK_WEBHOOK_URL")
        )

    def evaluate(self, report: DriftReport) -> List[AlertEvent]:
        """
        Check all metrics in *report* and trigger alerts where thresholds are exceeded.

        Args:
            report: DriftReport produced by DriftDetector.

        Returns:
            List of AlertEvent instances for each triggered alert.
        """
        events: List[AlertEvent] = []

        for metric in report.exceeded_metrics:
            event = self._build_event(metric)
            events.append(event)
            self._dispatch(event)

        if not events:
            logger.info("No drift alerts triggered for suite '%s'.", report.suite_name)
        else:
            logger.warning(
                "%d drift alert(s) triggered for suite '%s'.",
                len(events), report.suite_name,
            )

        return events

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_event(self, metric: DriftMetric) -> AlertEvent:
        """Build a human-readable AlertEvent from a DriftMetric."""
        message = (
            f"[DRIFT ALERT] {metric.drift_type.upper()} drift detected in "
            f"suite='{metric.suite_name}' test='{metric.test_name}': "
            f"score={metric.drift_score:.4f} (threshold={metric.threshold}). "
            f"{metric.details}"
        )
        return AlertEvent(
            suite_name=metric.suite_name,
            test_name=metric.test_name,
            drift_type=metric.drift_type,
            drift_score=metric.drift_score,
            threshold=metric.threshold,
            details=metric.details,
            message=message,
        )

    def _dispatch(self, event: AlertEvent) -> None:
        """Route an alert event to all configured channels."""
        # Always log to console
        logger.warning(event.message)

        # Optional: webhook
        if self._webhook_url:
            self._send_webhook(event)

        # Optional: Slack
        if self._slack_url:
            self._send_slack(event)

    def _send_webhook(self, event: AlertEvent) -> None:
        """POST the alert payload to the configured webhook URL."""
        try:
            import requests
            payload = {
                "suite_name": event.suite_name,
                "test_name": event.test_name,
                "drift_type": event.drift_type,
                "drift_score": event.drift_score,
                "threshold": event.threshold,
                "details": event.details,
                "message": event.message,
            }
            resp = requests.post(
                self._webhook_url,
                json=payload,
                timeout=10,
                headers={"Content-Type": "application/json"},
            )
            if not resp.ok:
                logger.error(
                    "Webhook alert delivery failed: HTTP %d", resp.status_code
                )
            else:
                logger.debug("Webhook alert delivered to %s", self._webhook_url)
        except Exception as exc:
            logger.error("Webhook alert delivery exception: %s", exc)

    def _send_slack(self, event: AlertEvent) -> None:
        """Send a Slack message via incoming webhook."""
        try:
            import requests
            payload = {
                "text": f":warning: *PromptBench Drift Alert*\n{event.message}",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": (
                                f":warning: *Drift Alert: {event.drift_type.upper()}*\n"
                                f"• Suite: `{event.suite_name}`\n"
                                f"• Test: `{event.test_name}`\n"
                                f"• Score: `{event.drift_score:.4f}` (threshold: `{event.threshold}`)\n"
                                f"• Details: {event.details}"
                            ),
                        },
                    }
                ],
            }
            resp = requests.post(self._slack_url, json=payload, timeout=10)
            if not resp.ok:
                logger.error("Slack alert delivery failed: HTTP %d", resp.status_code)
            else:
                logger.debug("Slack alert delivered.")
        except Exception as exc:
            logger.error("Slack alert delivery exception: %s", exc)


def check_and_alert(suite_name: str, window: int = 50, storage=None) -> List[AlertEvent]:
    """
    Convenience function: run drift detection and fire alerts for *suite_name*.

    Args:
        suite_name: Suite to analyze.
        window:     Number of recent production logs to use per test.
        storage:    Optional Storage instance (defaults to production storage).

    Returns:
        List of AlertEvent instances for triggered alerts.
    """
    from promptbench.monitoring.drift import DriftDetector

    detector = DriftDetector(storage=storage)
    report = detector.analyze_suite(suite_name, window=window)

    manager = AlertManager()
    return manager.evaluate(report)
