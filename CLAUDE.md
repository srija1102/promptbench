# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

You are an expert Python engineer. Build me a complete,
production-ready open source CLI tool called "promptbench".

## What it does
promptbench is a prompt regression testing framework for LLMs.
It lets developers define expected behaviors for their AI prompts
in YAML files, then automatically tests those prompts against any
LLM and fails if the outputs regress from a known baseline.

## Complete file structure to generate:

promptbench/
├── promptbench/
│   ├── __init__.py
│   ├── cli.py
│   ├── runner.py
│   ├── evaluators/
│   │   ├── __init__.py
│   │   ├── format_check.py
│   │   ├── length_check.py
│   │   ├── tone_check.py
│   │   ├── content_check.py
│   │   └── semantic_check.py
│   ├── storage.py
│   ├── reporter.py
│   └── config.py
├── tests/
│   ├── test_runner.py
│   ├── test_evaluators.py
│   └── fixtures/sample_suite.yaml
├── examples/
│   ├── summarizer.yaml
│   ├── classifier.yaml
│   └── chatbot.yaml
├── .github/workflows/promptbench.yml
├── pyproject.toml
└── README.md

## Detailed requirements for each file:

### cli.py
Use the Typer library. Implement these commands:
- `promptbench baseline --suite <path>`
  Runs the suite once and saves outputs as the baseline in SQLite
- `promptbench test --suite <path>`
  Runs the suite and compares against baseline, exits with code 1
  if any test fails
- `promptbench report --suite <path> --format [text|html]`
  Generates a full test report
- `promptbench list`
  Lists all registered suites and their last run status

### runner.py
- Load a YAML test suite file
- For each test in the suite, call the LLM (Groq API with
  llama3-8b-8192 model by default)
- Pass the result through all relevant evaluators
- Return a structured TestResult object with: test name,
  passed/failed, actual output, expected output, duration ms,
  failure reason if any

### evaluators/format_check.py
Check if output matches expected format:
- "bullet_points": output contains lines starting with -, *, or •
- "json": output is valid parseable JSON
- "numbered_list": output contains lines starting with 1. 2. 3.
- "markdown": output contains markdown syntax
- "plain_text": output has no special formatting

### evaluators/length_check.py
Check output length constraints:
- max_words: word count must not exceed this
- min_words: word count must be at least this
- max_bullets: number of bullet points must not exceed this
- min_bullets: number of bullet points must be at least this
- max_sentences: sentence count must not exceed this

### evaluators/content_check.py
Check content requirements:
- required_phrases: list of strings that MUST appear in output
- forbidden_phrases: list of strings that must NOT appear in output
- forbidden_patterns: list of regex patterns that must NOT match
- required_patterns: list of regex patterns that MUST match

### evaluators/semantic_check.py
Check semantic similarity against baseline:
- Use sentence-transformers with "all-MiniLM-L6-v2" model (free,
  runs locally)
- Compute cosine similarity between current output and baseline
  output
- Fail if similarity drops below threshold (default 0.75)
- This catches cases where meaning has drifted even if format
  is correct

### evaluators/tone_check.py
Simple tone/sentiment check:
- Use a basic rule-based approach (no external API needed)
- Detect if tone shifted from professional to casual
- Check for: excessive hedging ("I think", "maybe", "possibly"),
  refusal patterns ("I cannot", "I'm unable", "As an AI"),
  overly casual language ("hey", "yeah", "gonna")
- Flag these as tone violations if they appear when baseline
  had none

### storage.py
Use SQLite (stdlib, no dependencies):
- Table: baselines (suite_name, test_name, output, timestamp,
  model, prompt_hash)
- Table: runs (suite_name, run_id, timestamp, passed, failed,
  duration_ms)
- Table: results (run_id, test_name, passed, actual_output,
  failure_reason, duration_ms)
- Methods: save_baseline(), get_baseline(), save_run(),
  get_run_history()
- Database file stored at ~/.promptbench/db.sqlite

### reporter.py
Generate test reports:
- Text report: colored terminal output using Rich library
  - Green checkmark for pass, red X for fail
  - Show test name, duration, failure reason
  - Summary line: "X passed, Y failed in Z seconds"
- HTML report: self-contained HTML file with:
  - Summary stats at top
  - Table of all tests with status
  - For failures: side-by-side diff of expected vs actual

### config.py
- Load YAML test suite files
- Validate required fields (name, model, tests)
- Resolve environment variables (GROQ_API_KEY, OPENAI_API_KEY)
- Support both single suite file and directory of suite files

### YAML test suite format
Each suite file must support:
```yaml
name: string
model: string (e.g. "llama3-8b-8192")
api_provider: string ("groq" | "openai" | "anthropic")
prompt_file: string (path to .txt file with the prompt template)
  OR
prompt: string (inline prompt text)
baseline_model: string (optional, model used for baseline)

tests:
  - name: string
    input: string (prompt with {variable} placeholders)
    variables: dict (values for placeholders)
    expect:
      format: string (optional)
      max_words: int (optional)
      min_words: int (optional)
      max_bullets: int (optional)
      min_bullets: int (optional)
      required_phrases: list[string] (optional)
      forbidden_phrases: list[string] (optional)
      forbidden_patterns: list[string] regex (optional)
      semantic_threshold: float 0-1 (optional, default 0.75)
      tone: string ("professional"|"casual"|"neutral") (optional)
```

### pyproject.toml
- Package name: promptbench
- Version: 0.1.0
- Python >=3.9
- Dependencies: typer, pyyaml, requests, sentence-transformers,
  rich, sqlite3 (stdlib)
- Entry point: promptbench = "promptbench.cli:app"
- Include all necessary metadata for PyPI publishing

### README.md
Write a compelling README with:
1. One-line description
2. "The problem" section — 3 sentences on why prompt regression
   is painful
3. Quick install: pip install promptbench
4. 5-minute quickstart with real code example
5. Full YAML syntax reference
6. GitHub Actions integration example
7. "How it works" — brief technical explanation
8. Contributing guide
9. MIT license badge

### examples/summarizer.yaml
A complete, realistic example test suite for an article
summarizer prompt with 5 tests covering: format, length,
content requirements, forbidden phrases, and semantic threshold.

### examples/classifier.yaml
A complete, realistic example for a sentiment classifier
prompt with 4 tests.

### tests/test_runner.py and test_evaluators.py
Write proper pytest tests with at least 3 tests per evaluator
using mocked LLM responses. Aim for 80% coverage.

### .github/workflows/promptbench.yml
Complete GitHub Actions workflow that:
- Triggers on push and pull_request
- Tests against Python 3.9, 3.10, 3.11
- Runs pytest
- Runs promptbench test on the example suites

## Coding standards:
- Type hints everywhere
- Docstrings on all public methods
- Error messages must be human-readable and actionable
- Never crash silently — always show what went wrong and how
  to fix it
- All API keys loaded from environment variables only, never
  hardcoded
- Exit code 0 = all tests passed, exit code 1 = any test failed
  (this is what makes GitHub Actions work)

## Generate every file completely — no placeholders, no "add
your code here" comments. Every file must be immediately
runnable. Start with cli.py, then runner.py, then each
evaluator, then storage.py, reporter.py, config.py, then
the YAML examples, then pyproject.toml, then README.md,
then the test files.
