"""
dashboard/api.py — FastAPI REST API for the promptbench dashboard.

Start the server:
    uvicorn dashboard.api:app --reload --port 7860

Endpoints:
    GET  /runs          — List recent test runs (all suites or filtered by suite)
    GET  /metrics       — Aggregate pass/fail metrics per suite
    GET  /failures      — List recent failing test results
    GET  /drift         — Recent drift metrics
    POST /optimize      — Trigger prompt optimization for a suite
    GET  /health        — Health check
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure the project root is on sys.path when running from any directory
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, HTMLResponse
    from pydantic import BaseModel
except ImportError as _e:
    raise ImportError(
        "FastAPI is required for the dashboard. "
        "Install it with: pip install fastapi uvicorn"
    ) from _e

from promptbench.storage import Storage

app = FastAPI(
    title="PromptBench Dashboard API",
    description="REST API for the PromptBench AI Reliability Platform",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_storage = Storage()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class OptimizeRequest(BaseModel):
    suite_path: str
    no_semantic: bool = False


class IngestLogRequest(BaseModel):
    suite_name: str
    test_name: str
    prompt: str
    response: str
    model: str
    metadata: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "promptbench"}


@app.get("/runs")
def list_runs(
    suite: Optional[str] = Query(None, description="Filter by suite name"),
    limit: int = Query(20, ge=1, le=200),
) -> List[Dict[str, Any]]:
    """
    Return recent test runs.

    Args:
        suite: Optional suite name filter. Returns all suites if omitted.
        limit: Maximum number of runs to return.
    """
    if suite:
        suite_names = [suite]
    else:
        suite_names = _storage.list_suites()

    runs = []
    for name in suite_names:
        for run in _storage.get_run_history(name, limit=limit):
            runs.append({
                "suite_name": run.suite_name,
                "run_id": run.run_id,
                "timestamp": run.timestamp,
                "passed": run.passed,
                "failed": run.failed,
                "duration_ms": run.duration_ms,
                "total": run.passed + run.failed,
                "pass_rate": (
                    run.passed / (run.passed + run.failed)
                    if (run.passed + run.failed) > 0
                    else 0.0
                ),
            })

    # Sort newest first
    runs.sort(key=lambda r: r["timestamp"], reverse=True)
    return runs[:limit]


@app.get("/metrics")
def get_metrics() -> List[Dict[str, Any]]:
    """
    Return aggregate pass/fail metrics per suite (based on all stored runs).
    """
    suite_names = _storage.list_suites()
    metrics = []

    for name in suite_names:
        history = _storage.get_run_history(name, limit=100)
        if not history:
            continue

        total_passed = sum(r.passed for r in history)
        total_failed = sum(r.failed for r in history)
        total_tests = total_passed + total_failed
        pass_rate = total_passed / total_tests if total_tests > 0 else 0.0
        latest = history[0]

        metrics.append({
            "suite_name": name,
            "total_runs": len(history),
            "total_passed": total_passed,
            "total_failed": total_failed,
            "pass_rate": round(pass_rate, 4),
            "last_run_timestamp": latest.timestamp,
            "last_run_status": "pass" if latest.failed == 0 else "fail",
        })

    return metrics


@app.get("/failures")
def list_failures(
    suite: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
) -> List[Dict[str, Any]]:
    """
    Return recent failing test results.

    Args:
        suite: Optional suite name filter.
        limit: Max number of failures to return.
    """
    suite_names = [suite] if suite else _storage.list_suites()
    failures = []

    for name in suite_names:
        history = _storage.get_run_history(name, limit=10)
        for run in history:
            results = _storage.get_run_results(run.run_id)
            for r in results:
                if not r.passed:
                    failures.append({
                        "suite_name": name,
                        "run_id": run.run_id,
                        "run_timestamp": run.timestamp,
                        "test_name": r.test_name,
                        "failure_reason": r.failure_reason,
                        "actual_output": r.actual_output[:500],
                        "duration_ms": r.duration_ms,
                    })

    failures.sort(key=lambda f: f["run_timestamp"], reverse=True)
    return failures[:limit]


@app.get("/drift")
def get_drift(
    suite: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
) -> List[Dict[str, Any]]:
    """
    Return recent drift metrics.

    Args:
        suite: Optional suite name filter.
        limit: Max number of drift records.
    """
    suite_names = [suite] if suite else _storage.list_suites()
    all_metrics = []

    for name in suite_names:
        metrics = _storage.get_drift_metrics(name, limit=limit)
        all_metrics.extend(metrics)

    all_metrics.sort(key=lambda m: m.get("timestamp", 0), reverse=True)
    return all_metrics[:limit]


@app.post("/ingest")
def ingest_log(request: IngestLogRequest) -> Dict[str, str]:
    """
    Ingest a production prompt/response pair for drift monitoring.
    """
    from promptbench.monitoring.ingest import LogIngester
    ingester = LogIngester(storage=_storage)
    ingester.ingest(
        suite_name=request.suite_name,
        test_name=request.test_name,
        prompt=request.prompt,
        response=request.response,
        model=request.model,
        metadata=request.metadata,
    )
    return {"status": "ingested"}


@app.post("/optimize")
def optimize_suite(request: OptimizeRequest) -> Dict[str, Any]:
    """
    Trigger prompt optimization for all failing tests in a suite.

    Returns a list of OptimizationResult-like dicts.
    """
    from promptbench.config import load_suite, get_api_key
    from promptbench.llm_providers import get_provider
    from promptbench.runner import TestRunner
    from promptbench.optimization.optimizer import PromptOptimizer

    suite_path = Path(request.suite_path)
    if not suite_path.exists():
        raise HTTPException(status_code=404, detail=f"Suite file not found: {suite_path}")

    try:
        config = load_suite(suite_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to load suite: {exc}")

    try:
        api_key = get_api_key(config.api_provider)
        provider = get_provider(config.api_provider, api_key)
    except EnvironmentError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    runner = TestRunner(storage=_storage)
    results = runner.run_suite(
        config, mode="test", use_semantic=not request.no_semantic
    )
    failing = [r for r in results if not r.passed]

    if not failing:
        return {"message": "All tests pass — no optimization needed.", "results": []}

    optimizer = PromptOptimizer(provider=provider, model=config.model)
    opt_results = optimizer.optimize_suite(config, failing)

    return {
        "message": f"Optimized {len(opt_results)} failing test(s).",
        "results": [
            {
                "test_name": o.test_name,
                "original_failure": o.original_failure,
                "improved": o.improved,
                "best_prompt": o.best_prompt,
                "error": o.error,
                "variants": [{"prompt": v.prompt, "score": v.score} for v in o.variants],
            }
            for o in opt_results
        ],
    }


@app.get("/", response_class=HTMLResponse)
def serve_ui() -> str:
    """Serve the dashboard UI."""
    ui_path = Path(__file__).parent / "ui" / "index.html"
    if ui_path.exists():
        return ui_path.read_text(encoding="utf-8")
    return "<h1>PromptBench Dashboard</h1><p>UI not found. Place index.html in dashboard/ui/</p>"
