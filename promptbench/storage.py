"""
storage.py — SQLite-backed persistence for baselines and test run history.

Database location: ~/.promptbench/db.sqlite

Schema:
  baselines  — stores the golden output for each test in each suite
  runs       — one row per promptbench test run (aggregate stats)
  results    — one row per individual test result within a run
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional


# ---------------------------------------------------------------------------
# Database location
# ---------------------------------------------------------------------------

_DB_DIR = Path.home() / ".promptbench"
_DB_PATH = _DB_DIR / "db.sqlite"


# ---------------------------------------------------------------------------
# Dataclasses for stored records
# ---------------------------------------------------------------------------

@dataclass
class BaselineRecord:
    """A stored baseline output for a single test."""

    suite_name: str
    test_name: str
    output: str
    timestamp: float
    model: str
    prompt_hash: str


@dataclass
class RunRecord:
    """Aggregate stats for a single promptbench run."""

    suite_name: str
    run_id: str
    timestamp: float
    passed: int
    failed: int
    duration_ms: int


@dataclass
class ResultRecord:
    """Result for one test within a run."""

    run_id: str
    test_name: str
    passed: bool
    actual_output: str
    failure_reason: Optional[str]
    duration_ms: int


# ---------------------------------------------------------------------------
# Storage class
# ---------------------------------------------------------------------------

class Storage:
    """
    Manages all SQLite reads and writes for promptbench.

    Usage:
        storage = Storage()
        storage.save_baseline("my-suite", "test-1", "output text", "llama3-8b-8192", "prompt")
        baseline = storage.get_baseline("my-suite", "test-1")
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        """
        Initialize storage at the given path (defaults to ~/.promptbench/db.sqlite).

        Args:
            db_path: Override the database path (useful for testing).
        """
        self._db_path = db_path or _DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Context manager for connections
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield a SQLite connection with row_factory set."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Schema initialization
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create tables if they do not exist."""
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS baselines (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    suite_name  TEXT    NOT NULL,
                    test_name   TEXT    NOT NULL,
                    output      TEXT    NOT NULL,
                    timestamp   REAL    NOT NULL,
                    model       TEXT    NOT NULL,
                    prompt_hash TEXT    NOT NULL,
                    UNIQUE(suite_name, test_name)
                );

                CREATE TABLE IF NOT EXISTS runs (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    suite_name  TEXT    NOT NULL,
                    run_id      TEXT    NOT NULL UNIQUE,
                    timestamp   REAL    NOT NULL,
                    passed      INTEGER NOT NULL DEFAULT 0,
                    failed      INTEGER NOT NULL DEFAULT 0,
                    duration_ms INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS results (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id         TEXT    NOT NULL,
                    test_name      TEXT    NOT NULL,
                    passed         INTEGER NOT NULL,
                    actual_output  TEXT    NOT NULL,
                    failure_reason TEXT,
                    duration_ms    INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );
            """)

    # ------------------------------------------------------------------
    # Baseline operations
    # ------------------------------------------------------------------

    @staticmethod
    def _prompt_hash(prompt: str) -> str:
        """Return a short SHA-256 hash of *prompt* for change detection."""
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

    def save_baseline(
        self,
        suite_name: str,
        test_name: str,
        output: str,
        model: str,
        prompt: str,
    ) -> None:
        """
        Save (or overwrite) the baseline output for a test.

        Args:
            suite_name: Name of the test suite.
            test_name:  Name of the individual test.
            output:     The LLM output to store as baseline.
            model:      Model identifier (e.g. "llama3-8b-8192").
            prompt:     The full prompt sent to the model (used for hashing).
        """
        ph = self._prompt_hash(prompt)
        ts = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO baselines (suite_name, test_name, output, timestamp, model, prompt_hash)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(suite_name, test_name) DO UPDATE SET
                    output      = excluded.output,
                    timestamp   = excluded.timestamp,
                    model       = excluded.model,
                    prompt_hash = excluded.prompt_hash
                """,
                (suite_name, test_name, output, ts, model, ph),
            )

    def get_baseline(self, suite_name: str, test_name: str) -> Optional[BaselineRecord]:
        """
        Retrieve the baseline record for a test, or None if not found.

        Args:
            suite_name: Name of the test suite.
            test_name:  Name of the individual test.

        Returns:
            BaselineRecord or None.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM baselines WHERE suite_name=? AND test_name=?",
                (suite_name, test_name),
            ).fetchone()
        if row is None:
            return None
        return BaselineRecord(
            suite_name=row["suite_name"],
            test_name=row["test_name"],
            output=row["output"],
            timestamp=row["timestamp"],
            model=row["model"],
            prompt_hash=row["prompt_hash"],
        )

    def list_baselines(self, suite_name: str) -> List[BaselineRecord]:
        """Return all baselines stored for *suite_name*."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM baselines WHERE suite_name=? ORDER BY test_name",
                (suite_name,),
            ).fetchall()
        return [
            BaselineRecord(
                suite_name=r["suite_name"],
                test_name=r["test_name"],
                output=r["output"],
                timestamp=r["timestamp"],
                model=r["model"],
                prompt_hash=r["prompt_hash"],
            )
            for r in rows
        ]

    def delete_baselines(self, suite_name: str) -> int:
        """Delete all baselines for *suite_name*. Returns the number of rows deleted."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM baselines WHERE suite_name=?", (suite_name,)
            )
            return cursor.rowcount

    # ------------------------------------------------------------------
    # Run operations
    # ------------------------------------------------------------------

    def save_run(
        self,
        suite_name: str,
        passed: int,
        failed: int,
        duration_ms: int,
        results: Optional[List[ResultRecord]] = None,
    ) -> str:
        """
        Persist a completed test run and its individual results.

        Args:
            suite_name:  Name of the test suite.
            passed:      Number of passing tests.
            failed:      Number of failing tests.
            duration_ms: Total wall-clock duration of the run in milliseconds.
            results:     Optional list of per-test ResultRecord instances.

        Returns:
            The generated run_id (UUID string).
        """
        run_id = str(uuid.uuid4())
        ts = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (suite_name, run_id, timestamp, passed, failed, duration_ms)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (suite_name, run_id, ts, passed, failed, duration_ms),
            )
            if results:
                conn.executemany(
                    """
                    INSERT INTO results (run_id, test_name, passed, actual_output,
                                        failure_reason, duration_ms)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            run_id,
                            r.test_name,
                            int(r.passed),
                            r.actual_output,
                            r.failure_reason,
                            r.duration_ms,
                        )
                        for r in results
                    ],
                )
        return run_id

    def get_run_history(self, suite_name: str, limit: int = 20) -> List[RunRecord]:
        """
        Return the last *limit* runs for *suite_name*, newest first.

        Args:
            suite_name: Name of the test suite.
            limit:      Maximum number of records to return.

        Returns:
            List of RunRecord instances.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM runs
                WHERE suite_name=?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (suite_name, limit),
            ).fetchall()
        return [
            RunRecord(
                suite_name=r["suite_name"],
                run_id=r["run_id"],
                timestamp=r["timestamp"],
                passed=r["passed"],
                failed=r["failed"],
                duration_ms=r["duration_ms"],
            )
            for r in rows
        ]

    def get_run_results(self, run_id: str) -> List[ResultRecord]:
        """Return all per-test results for a given *run_id*."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM results WHERE run_id=? ORDER BY test_name",
                (run_id,),
            ).fetchall()
        return [
            ResultRecord(
                run_id=r["run_id"],
                test_name=r["test_name"],
                passed=bool(r["passed"]),
                actual_output=r["actual_output"],
                failure_reason=r["failure_reason"],
                duration_ms=r["duration_ms"],
            )
            for r in rows
        ]

    def list_suites(self) -> List[str]:
        """Return the names of all suites that have stored baselines or runs."""
        with self._connect() as conn:
            baseline_suites = {
                row[0]
                for row in conn.execute("SELECT DISTINCT suite_name FROM baselines").fetchall()
            }
            run_suites = {
                row[0]
                for row in conn.execute("SELECT DISTINCT suite_name FROM runs").fetchall()
            }
        return sorted(baseline_suites | run_suites)
