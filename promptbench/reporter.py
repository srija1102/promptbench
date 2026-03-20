"""
reporter.py — Generate text and HTML reports from test results.

Text reports use Rich for colored terminal output.
HTML reports are fully self-contained (no external CSS/JS dependencies).
"""

from __future__ import annotations

import difflib
import html as html_module
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text

from promptbench.runner import TestResult


# ---------------------------------------------------------------------------
# Text reporter
# ---------------------------------------------------------------------------

class TextReporter:
    """Render test results to the terminal using Rich."""

    def __init__(self, console: Optional[Console] = None) -> None:
        """
        Args:
            console: Rich Console instance (defaults to a new Console()).
        """
        self._console = console or Console()

    def render(
        self,
        suite_name: str,
        results: List[TestResult],
        total_duration_ms: int,
    ) -> None:
        """
        Print a full test report to the terminal.

        Args:
            suite_name:       Name of the test suite.
            results:          List of TestResult instances.
            total_duration_ms: Wall-clock time for the full run, in ms.
        """
        passed = [r for r in results if r.passed]
        failed = [r for r in results if not r.passed]

        self._console.print()
        self._console.print(
            Panel(
                f"[bold]promptbench[/bold] — [dim]{suite_name}[/dim]",
                expand=False,
                border_style="blue",
            )
        )

        table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
        table.add_column("", width=3)
        table.add_column("Test", style="bold", no_wrap=True)
        table.add_column("Duration", justify="right", style="dim", width=10)
        table.add_column("Failure Reason", overflow="fold")

        for result in results:
            if result.passed:
                status = Text("✓", style="bold green")
                reason_text = Text("—", style="dim")
            else:
                status = Text("✗", style="bold red")
                reason_text = Text(result.failure_reason or "unknown failure", style="red")

            table.add_row(
                status,
                result.test_name,
                f"{result.duration_ms}ms",
                reason_text,
            )

        self._console.print(table)

        # Summary line
        total = len(results)
        duration_s = total_duration_ms / 1000
        if failed:
            summary_style = "bold red"
            summary = (
                f"[bold red]{len(failed)} failed[/bold red], "
                f"[bold green]{len(passed)} passed[/bold green] "
                f"[dim]in {duration_s:.2f}s[/dim]"
            )
        else:
            summary_style = "bold green"
            summary = (
                f"[bold green]{len(passed)}/{total} passed[/bold green] "
                f"[dim]in {duration_s:.2f}s[/dim]"
            )

        self._console.print(summary)
        self._console.print()

        # Detail sections for failures
        if failed:
            self._console.print("[bold red]─── Failure Details ───[/bold red]")
            for result in failed:
                self._console.print()
                self._console.print(f"[bold red]✗ {result.test_name}[/bold red]")
                self._console.print(
                    f"  [dim]Reason:[/dim] {result.failure_reason or 'unknown'}"
                )
                if result.expected_output:
                    self._console.print(
                        f"  [dim]Baseline (first 200 chars):[/dim] "
                        f"{result.expected_output[:200]!r}"
                    )
                self._console.print(
                    f"  [dim]Actual   (first 200 chars):[/dim] "
                    f"{result.actual_output[:200]!r}"
                )
            self._console.print()


# ---------------------------------------------------------------------------
# HTML reporter
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>promptbench report — {suite_name}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #0f0f0f;
      color: #e4e4e4;
      margin: 0;
      padding: 24px;
    }}
    h1 {{ font-size: 1.5rem; margin-bottom: 4px; }}
    .meta {{ color: #888; font-size: 0.875rem; margin-bottom: 24px; }}
    .summary {{
      display: flex; gap: 16px; margin-bottom: 28px;
    }}
    .stat {{
      background: #1a1a1a;
      border: 1px solid #333;
      border-radius: 8px;
      padding: 16px 24px;
      text-align: center;
    }}
    .stat .number {{ font-size: 2rem; font-weight: 700; }}
    .stat .label  {{ font-size: 0.75rem; color: #888; text-transform: uppercase; }}
    .passed-num {{ color: #4ade80; }}
    .failed-num {{ color: #f87171; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.875rem;
      margin-bottom: 32px;
    }}
    th {{
      text-align: left;
      padding: 10px 12px;
      border-bottom: 2px solid #333;
      color: #888;
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    td {{
      padding: 10px 12px;
      border-bottom: 1px solid #222;
      vertical-align: top;
    }}
    .badge {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      font-weight: 600;
    }}
    .badge-pass {{ background: #14532d; color: #4ade80; }}
    .badge-fail {{ background: #450a0a; color: #f87171; }}
    .diff-section {{ margin-top: 40px; }}
    .diff-title {{
      font-size: 1rem; font-weight: 600; color: #f87171;
      margin-bottom: 12px;
    }}
    .diff-card {{
      background: #141414;
      border: 1px solid #333;
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 20px;
    }}
    .diff-card h3 {{ margin: 0 0 12px 0; font-size: 0.9rem; color: #e4e4e4; }}
    .diff-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .diff-box {{ background: #0a0a0a; border-radius: 4px; padding: 12px; }}
    .diff-box .diff-label {{ font-size: 0.7rem; text-transform: uppercase;
                              color: #888; margin-bottom: 8px; }}
    pre {{ margin: 0; white-space: pre-wrap; word-break: break-all;
           font-size: 0.8rem; line-height: 1.5; }}
    .diff-added   {{ background: #14532d33; }}
    .diff-removed {{ background: #450a0a33; }}
    .reason {{ color: #fbbf24; font-size: 0.8rem; }}
  </style>
</head>
<body>
  <h1>promptbench report</h1>
  <div class="meta">Suite: <strong>{suite_name}</strong> &nbsp;·&nbsp; {run_time}</div>

  <div class="summary">
    <div class="stat">
      <div class="number">{total}</div>
      <div class="label">Total Tests</div>
    </div>
    <div class="stat">
      <div class="number passed-num">{passed}</div>
      <div class="label">Passed</div>
    </div>
    <div class="stat">
      <div class="number failed-num">{failed}</div>
      <div class="label">Failed</div>
    </div>
    <div class="stat">
      <div class="number">{duration_s:.2f}s</div>
      <div class="label">Duration</div>
    </div>
  </div>

  <table>
    <thead>
      <tr>
        <th>Status</th>
        <th>Test Name</th>
        <th>Duration</th>
        <th>Failure Reason</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>

  {diff_section}
</body>
</html>
"""


def _build_table_rows(results: List[TestResult]) -> str:
    rows = []
    for r in results:
        badge = (
            '<span class="badge badge-pass">PASS</span>'
            if r.passed
            else '<span class="badge badge-fail">FAIL</span>'
        )
        reason = html_module.escape(r.failure_reason or "—") if not r.passed else "—"
        rows.append(
            f"<tr>"
            f"<td>{badge}</td>"
            f"<td>{html_module.escape(r.test_name)}</td>"
            f"<td>{r.duration_ms}ms</td>"
            f"<td class='reason'>{reason}</td>"
            f"</tr>"
        )
    return "\n      ".join(rows)


def _build_diff_section(results: List[TestResult]) -> str:
    failed = [r for r in results if not r.passed]
    if not failed:
        return ""

    cards = []
    for r in failed:
        baseline = r.expected_output or ""
        actual = r.actual_output or ""

        # Side-by-side diff using difflib
        diff_html_parts = []
        matcher = difflib.SequenceMatcher(None, baseline.split("\n"), actual.split("\n"))
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for line in baseline.split("\n")[i1:i2]:
                    diff_html_parts.append(html_module.escape(line) + "\n")
            elif tag in ("replace", "delete"):
                for line in baseline.split("\n")[i1:i2]:
                    diff_html_parts.append(
                        f'<span class="diff-removed">- {html_module.escape(line)}</span>\n'
                    )
            elif tag == "insert":
                for line in actual.split("\n")[j1:j2]:
                    diff_html_parts.append(
                        f'<span class="diff-added">+ {html_module.escape(line)}</span>\n'
                    )

        diff_content = "".join(diff_html_parts)

        card = f"""
    <div class="diff-card">
      <h3>✗ {html_module.escape(r.test_name)}</h3>
      <p class="reason">{html_module.escape(r.failure_reason or 'unknown failure')}</p>
      <div class="diff-grid">
        <div class="diff-box">
          <div class="diff-label">Baseline Output</div>
          <pre>{html_module.escape(baseline[:1000])}</pre>
        </div>
        <div class="diff-box">
          <div class="diff-label">Actual Output</div>
          <pre>{html_module.escape(actual[:1000])}</pre>
        </div>
      </div>
    </div>"""
        cards.append(card)

    return (
        '<div class="diff-section">'
        '<div class="diff-title">Failure Details (side-by-side diff)</div>'
        + "".join(cards)
        + "</div>"
    )


class HtmlReporter:
    """Render test results as a self-contained HTML file."""

    def render(
        self,
        suite_name: str,
        results: List[TestResult],
        total_duration_ms: int,
    ) -> str:
        """
        Build the full HTML report string.

        Args:
            suite_name:        Name of the test suite.
            results:           List of TestResult instances.
            total_duration_ms: Total wall-clock duration in ms.

        Returns:
            Complete HTML document as a string.
        """
        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed)
        total = len(results)
        duration_s = total_duration_ms / 1000
        run_time = datetime.utcfromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M UTC")

        return _HTML_TEMPLATE.format(
            suite_name=html_module.escape(suite_name),
            run_time=run_time,
            total=total,
            passed=passed,
            failed=failed,
            duration_s=duration_s,
            rows=_build_table_rows(results),
            diff_section=_build_diff_section(results),
        )

    def save(
        self,
        suite_name: str,
        results: List[TestResult],
        total_duration_ms: int,
        output_path: Path,
    ) -> None:
        """
        Write the HTML report to *output_path*.

        Args:
            suite_name:        Name of the test suite.
            results:           List of TestResult instances.
            total_duration_ms: Total wall-clock duration in ms.
            output_path:       Destination file path.
        """
        html_content = self.render(suite_name, results, total_duration_ms)
        output_path.write_text(html_content, encoding="utf-8")
