"""
monitoring/ingest.py — Ingest production LLM logs into the database.

Production logs are raw prompt/response pairs captured from your live system.
They are used by the drift detection engine to compare against baselines.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

from promptbench.storage import Storage

logger = logging.getLogger(__name__)


@dataclass
class ProductionLog:
    """A single production prompt/response pair."""

    suite_name: str
    test_name: str
    prompt: str
    response: str
    model: str
    timestamp: float = 0.0
    metadata: Optional[str] = None  # JSON string for arbitrary key-value pairs


class LogIngester:
    """
    Stores production LLM calls to the database for drift analysis.

    Usage:
        ingester = LogIngester()
        ingester.ingest(
            suite_name="chatbot",
            test_name="greeting",
            prompt="Hello, can you help me?",
            response="Of course! How can I assist you today?",
            model="llama3-8b-8192",
        )
    """

    def __init__(self, storage: Optional[Storage] = None) -> None:
        self._storage = storage or Storage()

    def ingest(
        self,
        suite_name: str,
        test_name: str,
        prompt: str,
        response: str,
        model: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Save a production log entry.

        Args:
            suite_name: Logical name of the suite / feature area.
            test_name:  Logical name of the test / prompt type.
            prompt:     The prompt sent to the LLM.
            response:   The LLM's response.
            model:      Model identifier.
            metadata:   Optional dict of extra context (serialized to JSON).
        """
        import json
        meta_str = json.dumps(metadata) if metadata else None
        log = ProductionLog(
            suite_name=suite_name,
            test_name=test_name,
            prompt=prompt,
            response=response,
            model=model,
            timestamp=time.time(),
            metadata=meta_str,
        )
        try:
            self._storage.save_production_log(log)
            logger.debug(
                "Ingested production log: suite=%s test=%s", suite_name, test_name
            )
        except Exception as exc:
            logger.error("Failed to ingest production log: %s", exc)

    def ingest_batch(self, logs: List[ProductionLog]) -> int:
        """
        Ingest multiple production logs at once.

        Args:
            logs: List of ProductionLog instances.

        Returns:
            Number of successfully ingested logs.
        """
        count = 0
        for log in logs:
            try:
                self._storage.save_production_log(log)
                count += 1
            except Exception as exc:
                logger.error(
                    "Failed to ingest log for %s/%s: %s",
                    log.suite_name, log.test_name, exc,
                )
        logger.info("Batch ingested %d/%d production logs.", count, len(logs))
        return count

    def get_recent_logs(
        self,
        suite_name: str,
        test_name: str,
        limit: int = 100,
    ) -> List[ProductionLog]:
        """
        Retrieve the most recent production logs for a suite/test combination.

        Args:
            suite_name: Suite to query.
            test_name:  Test to query.
            limit:      Maximum number of records.

        Returns:
            List of ProductionLog instances, newest first.
        """
        return self._storage.get_production_logs(suite_name, test_name, limit=limit)
