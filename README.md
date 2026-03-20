# promptbench

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![CI](https://github.com/your-org/promptbench/actions/workflows/promptbench.yml/badge.svg)](https://github.com/your-org/promptbench/actions)

**Prompt regression testing for LLMs — catch output regressions before they reach production.**

---

## The Problem

You carefully craft a prompt. It works beautifully. Then someone updates the system prompt, changes the model version, or tweaks a parameter — and silently, the outputs degrade. Responses that used to be bullet-point summaries now come back as paragraphs. JSON outputs gain extra commentary. The professional tone drifts toward refusal boilerplate.

Without automated tests, you discover these regressions in production. **promptbench** fixes that.

Define expected behaviors once in a YAML file. Run `promptbench baseline` to capture golden outputs. From then on, every CI run compares live LLM outputs against that baseline and fails loudly if anything regresses.

---

## Quick Install

```bash
pip install promptbench
```

You'll need a free [Groq API key](https://console.groq.com) (or OpenAI / Anthropic):

```bash
export GROQ_API_KEY=your_key_here
```

---

## 5-Minute Quickstart

### 1. Create a test suite

```yaml
# my_suite.yaml
name: my-summarizer
model: llama3-8b-8192
api_provider: groq

prompt: |
  You are a professional summarizer. Summarize the input in bullet points,
  under 100 words. Use a professional tone.

tests:
  - name: summarize-tech-article
    input: |
      Summarize this: AI is transforming medicine, finance, and transportation.
      Companies are investing billions. Experts predict 30% job automation by 2030.
    expect:
      format: bullet_points
      max_words: 100
      min_bullets: 2
      tone: professional
      forbidden_phrases:
        - "As an AI"
        - "I cannot"
```

### 2. Save the baseline

```bash
promptbench baseline --suite my_suite.yaml
```

This calls the LLM once and stores the outputs as your golden baseline.

### 3. Run tests (now and forever)

```bash
promptbench test --suite my_suite.yaml
```

```
✓ summarize-tech-article    342ms  —

1/1 passed in 0.34s
```

Change your prompt and re-run — promptbench tells you exactly what broke:

```
✗ summarize-tech-article    287ms  Expected bullet-point list but none found.

1 failed, 0 passed in 0.29s
```

### 4. Generate a report

```bash
# Terminal (colored)
promptbench report --suite my_suite.yaml --format text

# Self-contained HTML with side-by-side diffs
promptbench report --suite my_suite.yaml --format html --output report.html
```

### 5. See all your suites

```bash
promptbench list
```

---

## Full YAML Syntax Reference

```yaml
# ── Required fields ───────────────────────────────────────────────────────
name: string                   # Unique suite identifier
model: string                  # Model ID (e.g. "llama3-8b-8192", "gpt-4o-mini")
api_provider: groq             # groq | openai | anthropic

# ── System prompt (one of the two) ────────────────────────────────────────
prompt: |                      # Inline system prompt
  You are a helpful assistant.

prompt_file: prompts/sys.txt   # Or load from a .txt file

# ── Optional ───────────────────────────────────────────────────────────────
baseline_model: gpt-4o         # Compare against a different model for baseline

# ── Test cases ─────────────────────────────────────────────────────────────
tests:
  - name: string               # Required. Unique within suite.
    input: string              # Required. User message. Supports {variable} placeholders.
    variables:                 # Optional. Dict of {placeholder: value} substitutions.
      topic: "climate change"

    expect:
      # Format checks
      format: bullet_points    # bullet_points | json | numbered_list | markdown | plain_text

      # Length checks
      max_words: 150           # Hard upper limit on total word count
      min_words: 10            # Hard lower limit on total word count
      max_bullets: 6           # Max number of bullet/list items
      min_bullets: 2           # Min number of bullet/list items
      max_sentences: 5         # Max sentences (approximate heuristic)

      # Content checks
      required_phrases:        # These strings MUST appear in output (case-insensitive)
        - "climate"
        - "temperature"
      forbidden_phrases:       # These strings must NOT appear (case-insensitive)
        - "As an AI"
        - "I cannot"
      required_patterns:       # Regex patterns that MUST match
        - '"sentiment"\s*:\s*"positive"'
      forbidden_patterns:      # Regex patterns that must NOT match
        - "(?i)I (cannot|can't|am unable)"

      # Semantic similarity (requires baseline)
      semantic_threshold: 0.75 # Cosine similarity threshold [0–1]. Default: 0.75

      # Tone check (rule-based, no API)
      tone: professional       # professional | casual | neutral
```

### Environment variable substitution

Any value in a YAML suite can reference environment variables:

```yaml
model: ${MY_CUSTOM_MODEL}
prompt: "Key: ${SOME_VALUE}"
```

---

## GitHub Actions Integration

Add this to `.github/workflows/promptbench.yml`:

```yaml
name: Prompt Regression

on: [push, pull_request]

jobs:
  prompt-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install promptbench
        run: pip install promptbench

      - name: Save baseline on main
        if: github.ref == 'refs/heads/main'
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
        run: promptbench baseline --suite prompts/my_suite.yaml

      - name: Test on PRs
        if: github.event_name == 'pull_request'
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
        run: promptbench test --suite prompts/my_suite.yaml --no-semantic
```

Add your `GROQ_API_KEY` in: **Repository Settings → Secrets → Actions**.

---

## How It Works

### Architecture

```
promptbench test --suite my_suite.yaml
        │
        ├── config.py        Load & validate YAML, resolve env vars
        ├── runner.py        Build prompts, call LLM, run evaluators
        │   ├── format_check.py   Structural format validation
        │   ├── length_check.py   Word/bullet/sentence counts
        │   ├── content_check.py  Phrase and regex matching
        │   ├── tone_check.py     Rule-based tone detection
        │   └── semantic_check.py Cosine similarity (sentence-transformers)
        ├── storage.py       SQLite persistence (~/.promptbench/db.sqlite)
        └── reporter.py      Rich terminal output + self-contained HTML
```

### Evaluators

| Evaluator | Method | External API? |
|---|---|---|
| Format | Regex pattern matching | No |
| Length | Word/sentence counting | No |
| Content | String search + regex | No |
| Tone | Rule-based pattern matching | No |
| Semantic | `all-MiniLM-L6-v2` (local) | No — runs locally |

The semantic evaluator downloads the `all-MiniLM-L6-v2` model (~80MB) on first use via `sentence-transformers`. All subsequent runs use the cached model. Use `--no-semantic` to skip this in fast CI runs.

### Storage

All baselines and run history are stored in `~/.promptbench/db.sqlite` (SQLite, no server required). Three tables:

- `baselines` — golden outputs keyed by suite + test name
- `runs` — aggregate stats per run (pass/fail counts, duration)
- `results` — per-test detail for every run

---

## CLI Reference

```
promptbench baseline --suite <path>
    Run the suite once and save outputs as the golden baseline.
    Use after intentional prompt changes.

promptbench test --suite <path> [--no-semantic]
    Run the suite against baseline. Exits 1 if any test fails.

promptbench report --suite <path> --format [text|html] [--output <file>]
    Generate a report. HTML includes side-by-side diffs.

promptbench list
    Show all suites with last-run status.
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Install in dev mode: `pip install -e ".[dev]"`
4. Write tests — at minimum, add cases to `tests/test_evaluators.py`
5. Run: `pytest tests/ -v`
6. Submit a pull request

### Adding a new evaluator

1. Create `promptbench/evaluators/my_evaluator.py` with a `MyEvaluator` class
2. Expose a `check(output, expect) -> Tuple[bool, Optional[str]]` method
3. Import and wire it in `runner.py`
4. Export from `evaluators/__init__.py`
5. Add tests to `tests/test_evaluators.py`

---

## License

MIT — see [LICENSE](LICENSE) for details.
