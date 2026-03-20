"""
cli.py — Typer-based command-line interface for promptbench.

Commands:
  baseline  — Run suite once, save outputs as golden baseline
  test      — Run suite against baseline, exit 1 on any failure
  report    — Generate text or HTML report for a suite
  list      — Show all registered suites and their last-run status
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
) -> None:
    """
    Run the suite once and save LLM outputs as the golden baseline.

    Run this command whenever you intentionally update your prompts and want
    to accept the new outputs as the new expected behavior.
    """
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
                # Run one test at a time by temporarily replacing the test list
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
    _console.print(
        f"[dim]Stored at ~/.promptbench/db.sqlite[/dim]"
    )


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
) -> None:
    """
    Run the suite and compare outputs against the stored baseline.

    Exits with code 1 if any test fails. Use this in CI pipelines to
    catch prompt regressions automatically.
    """
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
) -> None:
    """
    Run the suite and generate a full report.

    For HTML reports, opens a self-contained .html file with side-by-side
    diffs for any failing tests.
    """
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
def list_suites() -> None:
    """
    List all registered suites and their last-run status.

    Shows suite names, number of tests run, pass/fail counts, and when
    the suite was last executed.
    """
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
