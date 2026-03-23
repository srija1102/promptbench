"""
cli.py — Typer-based command-line interface for promptbench.

Commands:
  baseline  — Run suite once, save outputs as golden baseline
  test      — Run suite against baseline, exit 1 on any failure
  report    — Generate text or HTML report for a suite
  list      — Show all registered suites and their last-run status
  optimize  — Automatically improve failing prompts using LLM
  monitor   — Detect behavioral drift in production logs and alert
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import box

from promptbench.config import load_suite
from promptbench.runner import TestRunner
from promptbench.reporter import TextReporter, HtmlReporter
from promptbench.storage import Storage


app = typer.Typer(
    name="promptbench",
    help="Prompt regression testing for LLMs.",
    add_completion=False,
    no_args_is_help=True,
)

_console = Console()
_err_console = Console(stderr=True, style="bold red")


# ---------------------------------------------------------------------------
# Shared verbose callback
# ---------------------------------------------------------------------------

def _setup_verbose(verbose: bool) -> None:
    if verbose:
        from promptbench.utils.logging import setup_logging
        setup_logging(verbose=True)


# ---------------------------------------------------------------------------
# baseline
# ---------------------------------------------------------------------------

@app.command()
def baseline(
    suite: Path = typer.Option(
        ...,
        "--suite",
        "-s",
        help="Path to the YAML test suite file.",
        exists=True,
        readable=True,
        resolve_path=True,
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging."),
) -> None:
    """
    Run the suite once and save LLM outputs as the golden baseline.

    Run this command whenever you intentionally update your prompts and want
    to accept the new outputs as the new expected behavior.
    """
    _setup_verbose(verbose)
    _console.print(f"[bold blue]promptbench baseline[/bold blue] — [dim]{suite}[/dim]")

    try:
        config = load_suite(suite)
    except Exception as exc:
        _err_console.print(f"Failed to load suite: {exc}")
        raise typer.Exit(code=1)

    _console.print(
        f"Suite [bold]{config.name}[/bold] · "
        f"{len(config.tests)} test(s) · "
        f"model [dim]{config.model}[/dim]"
    )

    runner = TestRunner()
    start = int(time.monotonic() * 1000)

    results = []
    for i, test in enumerate(config.tests, 1):
        _console.print(f"  [{i}/{len(config.tests)}] Saving baseline for [dim]{test.name}[/dim]...")
        try:
            result_list = runner.run_suite(
                _suite_with_single_test(config, test),
                mode="baseline",
            )
            results.extend(result_list)
        except Exception as exc:
            _err_console.print(f"  Error on test '{test.name}': {exc}")
            raise typer.Exit(code=1)

    total_ms = int(time.monotonic() * 1000) - start
    _console.print(
        f"\n[bold green]✓ Baseline saved[/bold green] — "
        f"{len(results)} test(s) in {total_ms / 1000:.2f}s"
    )
    _console.print(f"[dim]Stored at ~/.promptbench/db.sqlite[/dim]")


# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------

@app.command()
def test(
    suite: Path = typer.Option(
        ...,
        "--suite",
        "-s",
        help="Path to the YAML test suite file.",
        exists=True,
        readable=True,
        resolve_path=True,
    ),
    no_semantic: bool = typer.Option(
        False,
        "--no-semantic",
        help="Skip semantic similarity checks (faster, no model download).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging."),
) -> None:
    """
    Run the suite and compare outputs against the stored baseline.

    Exits with code 1 if any test fails. Use this in CI pipelines to
    catch prompt regressions automatically.
    """
    _setup_verbose(verbose)
    _console.print(f"[bold blue]promptbench test[/bold blue] — [dim]{suite}[/dim]")

    try:
        config = load_suite(suite)
    except Exception as exc:
        _err_console.print(f"Failed to load suite: {exc}")
        raise typer.Exit(code=1)

    _console.print(
        f"Suite [bold]{config.name}[/bold] · "
        f"{len(config.tests)} test(s) · "
        f"model [dim]{config.model}[/dim]"
    )

    runner = TestRunner()
    start = int(time.monotonic() * 1000)

    try:
        results = runner.run_suite(config, mode="test", use_semantic=not no_semantic)
    except EnvironmentError as exc:
        _err_console.print(str(exc))
        raise typer.Exit(code=1)
    except Exception as exc:
        _err_console.print(f"Suite run failed: {exc}")
        raise typer.Exit(code=1)

    total_ms = int(time.monotonic() * 1000) - start
    runner.save_run(config.name, results, total_ms)

    reporter = TextReporter(console=_console)
    reporter.render(config.name, results, total_ms)

    failed_count = sum(1 for r in results if not r.passed)
    if failed_count > 0:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

@app.command()
def report(
    suite: Path = typer.Option(
        ...,
        "--suite",
        "-s",
        help="Path to the YAML test suite file.",
        exists=True,
        readable=True,
        resolve_path=True,
    ),
    fmt: str = typer.Option(
        "text",
        "--format",
        "-f",
        help="Output format: 'text' (terminal) or 'html' (file).",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path for HTML reports (defaults to <suite_name>_report.html).",
    ),
    no_semantic: bool = typer.Option(
        False,
        "--no-semantic",
        help="Skip semantic similarity checks.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging."),
) -> None:
    """
    Run the suite and generate a full report.

    For HTML reports, opens a self-contained .html file with side-by-side
    diffs for any failing tests.
    """
    _setup_verbose(verbose)
    if fmt not in ("text", "html"):
        _err_console.print(f"Unknown format '{fmt}'. Use 'text' or 'html'.")
        raise typer.Exit(code=1)

    try:
        config = load_suite(suite)
    except Exception as exc:
        _err_console.print(f"Failed to load suite: {exc}")
        raise typer.Exit(code=1)

    runner = TestRunner()
    start = int(time.monotonic() * 1000)

    try:
        results = runner.run_suite(config, mode="test", use_semantic=not no_semantic)
    except EnvironmentError as exc:
        _err_console.print(str(exc))
        raise typer.Exit(code=1)
    except Exception as exc:
        _err_console.print(f"Suite run failed: {exc}")
        raise typer.Exit(code=1)

    total_ms = int(time.monotonic() * 1000) - start

    if fmt == "text":
        reporter = TextReporter(console=_console)
        reporter.render(config.name, results, total_ms)
    else:
        html_reporter = HtmlReporter()
        out_path = output or Path(f"{config.name.replace(' ', '_')}_report.html")
        html_reporter.save(config.name, results, total_ms, out_path)
        _console.print(f"[bold green]✓ HTML report written to:[/bold green] {out_path.resolve()}")

    failed_count = sum(1 for r in results if not r.passed)
    if failed_count > 0:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

@app.command(name="list")
def list_suites(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging."),
) -> None:
    """
    List all registered suites and their last-run status.

    Shows suite names, number of tests run, pass/fail counts, and when
    the suite was last executed.
    """
    _setup_verbose(verbose)
    storage = Storage()
    suite_names = storage.list_suites()

    if not suite_names:
        _console.print(
            "[dim]No suites found. Run 'promptbench baseline --suite <path>' to get started.[/dim]"
        )
        return

    table = Table(
        title="Registered Suites",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold",
    )
    table.add_column("Suite Name", style="bold")
    table.add_column("Last Run", style="dim")
    table.add_column("Passed", justify="right", style="green")
    table.add_column("Failed", justify="right", style="red")
    table.add_column("Duration", justify="right", style="dim")
    table.add_column("Status")

    from datetime import datetime

    for name in suite_names:
        history = storage.get_run_history(name, limit=1)
        if history:
            run = history[0]
            last_run = datetime.utcfromtimestamp(run.timestamp).strftime("%Y-%m-%d %H:%M")
            duration = f"{run.duration_ms / 1000:.2f}s"
            passed_str = str(run.passed)
            failed_str = str(run.failed)
            if run.failed == 0:
                status = "[bold green]PASS[/bold green]"
            else:
                status = f"[bold red]FAIL ({run.failed} failed)[/bold red]"
        else:
            last_run = "never"
            duration = "—"
            passed_str = "—"
            failed_str = "—"
            status = "[dim]baseline only[/dim]"

        table.add_row(name, last_run, passed_str, failed_str, duration, status)

    _console.print(table)


# ---------------------------------------------------------------------------
# optimize
# ---------------------------------------------------------------------------

@app.command()
def optimize(
    suite: Path = typer.Option(
        ...,
        "--suite",
        "-s",
        help="Path to the YAML test suite file.",
        exists=True,
        readable=True,
        resolve_path=True,
    ),
    no_semantic: bool = typer.Option(
        False,
        "--no-semantic",
        help="Skip semantic similarity checks.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging."),
) -> None:
    """
    Automatically improve failing prompts using LLM-generated variants.

    Runs the suite, identifies failing tests, generates 3 improved prompt
    variants per failure, re-evaluates each, and prints the best variant.
    """
    _setup_verbose(verbose)
    _console.print(f"[bold blue]promptbench optimize[/bold blue] — [dim]{suite}[/dim]")

    try:
        config = load_suite(suite)
    except Exception as exc:
        _err_console.print(f"Failed to load suite: {exc}")
        raise typer.Exit(code=1)

    # Run suite to find failures
    runner = TestRunner()
    try:
        results = runner.run_suite(config, mode="test", use_semantic=not no_semantic)
    except EnvironmentError as exc:
        _err_console.print(str(exc))
        raise typer.Exit(code=1)
    except Exception as exc:
        _err_console.print(f"Suite run failed: {exc}")
        raise typer.Exit(code=1)

    failing = [r for r in results if not r.passed]
    if not failing:
        _console.print("[bold green]✓ All tests pass — no optimization needed.[/bold green]")
        return

    _console.print(
        f"\n[yellow]{len(failing)} failing test(s) found. Running optimizer...[/yellow]\n"
    )

    # Set up provider and optimizer
    try:
        from promptbench.config import get_api_key
        from promptbench.llm_providers import get_provider
        from promptbench.optimization.optimizer import PromptOptimizer

        api_key = get_api_key(config.api_provider)
        provider = get_provider(config.api_provider, api_key)
        optimizer = PromptOptimizer(provider=provider, model=config.model)
    except EnvironmentError as exc:
        _err_console.print(str(exc))
        raise typer.Exit(code=1)
    except Exception as exc:
        _err_console.print(f"Failed to initialize optimizer: {exc}")
        raise typer.Exit(code=1)

    opt_results = optimizer.optimize_suite(config, failing)

    any_improved = False
    for opt in opt_results:
        _console.print(f"[bold]Test:[/bold] {opt.test_name}")
        _console.print(f"  [dim]Original failure:[/dim] {opt.original_failure}")

        if opt.error:
            _console.print(f"  [red]Optimization error:[/red] {opt.error}")
        elif opt.improved:
            any_improved = True
            _console.print(f"  [bold green]✓ Improved prompt found:[/bold green]")
            _console.print(f"  [dim]{opt.best_prompt}[/dim]")
        else:
            _console.print(f"  [yellow]No improvement found. Best candidate:[/yellow]")
            if opt.best_prompt:
                _console.print(f"  [dim]{opt.best_prompt}[/dim]")

        _console.print()

    if any_improved:
        _console.print(
            "[bold green]Optimization complete.[/bold green] "
            "Update your YAML suite with the improved prompts above."
        )
    else:
        _console.print(
            "[yellow]Optimization ran but found no improvements. "
            "Try adjusting your test expectations or using a stronger model.[/yellow]"
        )
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# monitor
# ---------------------------------------------------------------------------

@app.command()
def monitor(
    suite_name: str = typer.Argument(
        ...,
        help="Name of the suite to monitor (as saved in the database).",
    ),
    window: int = typer.Option(
        50,
        "--window",
        "-w",
        help="Number of recent production logs to analyze per test.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging."),
) -> None:
    """
    Detect behavioral drift in production LLM logs and fire alerts.

    Analyzes recent production logs ingested via the monitoring API and
    compares them against stored baselines to detect semantic, tone, and
    length drift. Exits with code 1 if any drift threshold is exceeded.
    """
    _setup_verbose(verbose)
    _console.print(
        f"[bold blue]promptbench monitor[/bold blue] — suite: [bold]{suite_name}[/bold]"
    )

    try:
        from promptbench.monitoring.drift import DriftDetector
        from promptbench.monitoring.alerts import AlertManager

        storage = Storage()
        detector = DriftDetector(storage=storage)
        report = detector.analyze_suite(suite_name, window=window)

    except Exception as exc:
        _err_console.print(f"Drift detection failed: {exc}")
        raise typer.Exit(code=1)

    if not report.metrics:
        _console.print(
            "[dim]No production logs found. "
            "Ingest logs first using the monitoring API or LogIngester.[/dim]"
        )
        return

    # Print drift table
    table = Table(
        title=f"Drift Report — {suite_name}",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold",
    )
    table.add_column("Test", style="bold")
    table.add_column("Type", style="dim")
    table.add_column("Score", justify="right")
    table.add_column("Threshold", justify="right", style="dim")
    table.add_column("Status")

    for m in report.metrics:
        if m.exceeded:
            status = "[bold red]DRIFT[/bold red]"
            score_str = f"[red]{m.drift_score:.4f}[/red]"
        else:
            status = "[bold green]OK[/bold green]"
            score_str = f"[green]{m.drift_score:.4f}[/green]"

        table.add_row(
            m.test_name,
            m.drift_type,
            score_str,
            str(m.threshold),
            status,
        )

    _console.print(table)

    # Fire alerts
    manager = AlertManager()
    alerts = manager.evaluate(report)

    if alerts:
        _console.print(
            f"\n[bold red]⚠ {len(alerts)} drift alert(s) triggered.[/bold red]"
        )
        for alert in alerts:
            _console.print(f"  [red]•[/red] {alert.message}")
        raise typer.Exit(code=1)
    else:
        _console.print(
            f"\n[bold green]✓ No drift detected across {len(report.metrics)} metric(s).[/bold green]"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _suite_with_single_test(suite, test):
    """Return a shallow copy of suite with only the given test."""
    import copy
    s = copy.copy(suite)
    s.tests = [test]
    return s


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point called by the 'promptbench' console script."""
    app()


if __name__ == "__main__":
    main()
