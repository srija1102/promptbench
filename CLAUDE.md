# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

You are a senior staff engineer, open-source maintainer, and startup CTO specializing in AI infrastructure and developer platforms.

Your task is to build a production-grade, open-source AI Reliability Platform called:

Your goal is to build a **complete, production-grade, open-source AI Reliability Platform** called:

# 🚀 PromptBench — Datadog for LLM Behavior

This must be:

* Fully functional
* Modular and extensible
* Safe, robust, and bug-resistant
* Developer-friendly
* Immediately runnable

You must implement EVERY file completely with NO placeholders.

---

# 🎯 PRODUCT PURPOSE

PromptBench allows developers to:

1. Define expected LLM behavior using YAML
2. Run prompts against LLMs
3. Detect regressions vs baseline
4. Evaluate output quality using rule-based checks + LLM-as-judge
5. Automatically improve prompts (optimization engine)
6. Monitor production behavior (drift detection)
7. Visualize via CLI + dashboard

---

# 📁 PROJECT STRUCTURE (STRICT — DO NOT CHANGE)

promptbench/
├── promptbench/
│   ├── **init**.py
│   ├── cli.py
│   ├── runner.py
│   ├── llm_providers.py
│   ├── config.py
│   ├── storage.py
│   ├── reporter.py
│   ├── utils/
│   │   ├── logging.py
│   │   └── helpers.py
│   ├── evaluators/
│   │   ├── format_check.py
│   │   ├── length_check.py
│   │   ├── content_check.py
│   │   ├── tone_check.py
│   │   ├── semantic_check.py
│   │   └── llm_judge.py
│   ├── optimization/
│   │   └── optimizer.py
│   ├── monitoring/
│   │   ├── drift.py
│   │   ├── ingest.py
│   │   └── alerts.py
├── dashboard/
│   ├── api.py
│   └── ui/index.html
├── examples/
├── tests/
├── .github/workflows/promptbench.yml
├── pyproject.toml
├── README.md
└── LICENSE

---

# 🧩 FILE-BY-FILE DETAILED REQUIREMENTS

---

# 1. cli.py (ENTRYPOINT)

Purpose:

* User interface for entire system

Must:

* Use Typer
* Define commands:

  * baseline
  * test
  * report
  * list
  * optimize
  * monitor

Each command must:

* Validate inputs
* Call appropriate modules (runner, storage, reporter, optimizer, monitoring)
* Handle exceptions cleanly
* Exit with:

  * 0 = success
  * 1 = failure

Include:

* --verbose flag (enables DEBUG logging)

---

# 2. config.py

Purpose:

* Load and validate YAML test suite files

Functions:

* load_config(path: str) -> List[SuiteConfig]
* validate_config(data: dict) -> None

Must:

* Support file OR directory input
* Validate required fields:

  * name
  * model
  * tests
* Validate:

  * regex patterns
  * numeric ranges (0–1)
* Resolve environment variables safely
* Provide clear error messages

---

# 3. llm_providers.py

Purpose:

* Abstract all LLM calls

Classes:

* LLMProvider (base class)
* GroqProvider
* OpenAIProvider
* AnthropicProvider

Methods:

* generate(prompt: str, model: str) -> str

Must include:

* retry logic (max 3, exponential backoff)
* timeout handling
* API key validation
* cost estimation (tokens approximation)
* consistent interface

---

# 4. runner.py (CORE ENGINE)

Purpose:

* Execute tests end-to-end

Functions:

* run_suite(config: SuiteConfig) -> List[TestResult]
* run_test(test_config) -> TestResult

Must:

* Resolve variables in prompts
* Call LLM provider
* Run ALL evaluators
* Support concurrency (ThreadPoolExecutor)
* Track:

  * execution time
  * cost
* Catch errors per test (no full crash)

Return:
TestResult dataclass with:

* name
* passed
* actual_output
* baseline_output
* duration_ms
* failure_reason
* cost
* llm_score

---

# 5. evaluators/

Each file implements ONE evaluator.

All evaluators must:

* follow common interface
* return EvaluationResult

---

## format_check.py

Check:

* bullet_points
* json
* numbered_list
* markdown
* plain_text

---

## length_check.py

Check:

* word count
* bullet count
* sentence count

---

## content_check.py

Check:

* required_phrases
* forbidden_phrases
* regex patterns

---

## tone_check.py

Detect:

* hedging
* refusal
* casual tone

Compare with baseline

---

## semantic_check.py

* Use sentence-transformers
* Compute cosine similarity
* Cache embeddings
* Fail below threshold

---

## llm_judge.py (CRITICAL)

Purpose:

* Use LLM to evaluate output quality

Steps:

1. Construct evaluation prompt
2. Call LLM provider
3. Parse JSON response safely
4. Extract:

   * score (1–5)
   * reason

Rules:

* If parsing fails:

  * retry once
  * else fail safely

---

# 6. storage.py

Purpose:

* Persist data

Functions:

* init_db()
* save_baseline()
* get_baseline()
* save_run()
* get_run_history()

Tables:

* baselines
* runs
* results
* production_logs
* drift_metrics

Must:

* auto-create DB
* handle DB errors safely

---

# 7. reporter.py

Purpose:

* Display results

Functions:

* generate_text_report(results)
* generate_html_report(results)

TEXT:

* Rich formatting
* PASS/FAIL colors

HTML:

* summary stats
* results table
* diff view

---

# 8. optimization/optimizer.py

Purpose:

* Improve prompts automatically

Steps:

1. Take failing prompt
2. Generate 3 improved prompts using LLM
3. Re-run evaluation
4. Select best

Must:

* return structured output
* handle failures safely

---

# 9. monitoring/

## drift.py

* compute embeddings
* detect:

  * semantic drift
  * tone drift
  * length drift

## ingest.py

* store production logs

## alerts.py

* trigger alerts when drift exceeds threshold

---

# 10. dashboard/api.py

Purpose:

* Provide REST API

Endpoints:

* /runs
* /metrics
* /failures
* /drift
* /optimize

Use FastAPI

---

# 11. dashboard/ui/index.html

Purpose:

* Simple frontend

Display:

* metrics
* failures
* drift alerts

---

# 🔒 SAFETY & QUALITY RULES

---

## ERROR HANDLING

Handle:

* API failures
* invalid YAML
* evaluator crashes
* DB issues

Never crash entire system

---

## VALIDATION

* strict schema validation
* regex validation
* numeric validation

---

## LOGGING

* INFO: normal flow
* ERROR: failures
* DEBUG: verbose

---

## CONCURRENCY

* thread-safe execution
* no shared mutable state

---

## NO SILENT FAILURES

* always report errors clearly

---

## USER-FRIENDLY ERRORS

Example:
"Invalid YAML: missing 'tests' field"

---

# 🧪 TESTING REQUIREMENTS

* pytest
* mock all LLM calls
* test:

  * each evaluator
  * runner
  * llm_judge parsing
  * optimizer
  * drift detection

---

# ⚙️ GITHUB ACTIONS

* run pytest
* run promptbench test
* fail on regression

---

# 📘 README.md

Must include:

* strong hook
* problem explanation
* quickstart
* examples
* screenshots (describe)

---

# 🧠 FINAL EXECUTION ORDER

Generate in this order:

1. cli.py
2. config.py
3. llm_providers.py
4. evaluators
5. runner.py
6. storage.py
7. reporter.py
8. optimization
9. monitoring
10. dashboard
11. examples
12. tests
13. GitHub Actions
14. README

---

# FINAL REQUIREMENT

The system must:

* install via pip install -e .
* run without crashing
* handle invalid inputs safely
* be usable immediately

---

Now build PromptBench completely.
