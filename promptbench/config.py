"""
config.py — Load and validate YAML test suite configurations.

Handles environment variable resolution, schema validation, and
multi-file suite discovery.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ExpectConfig:
    """Expected behavior constraints for a single test."""

    format: Optional[str] = None                    # bullet_points|json|numbered_list|markdown|plain_text
    max_words: Optional[int] = None
    min_words: Optional[int] = None
    max_bullets: Optional[int] = None
    min_bullets: Optional[int] = None
    max_sentences: Optional[int] = None
    required_phrases: List[str] = field(default_factory=list)
    forbidden_phrases: List[str] = field(default_factory=list)
    forbidden_patterns: List[str] = field(default_factory=list)
    required_patterns: List[str] = field(default_factory=list)
    semantic_threshold: float = 0.75
    tone: Optional[str] = None                      # professional|casual|neutral


@dataclass
class TestCase:
    """A single test case within a suite."""

    name: str
    input: str
    variables: Dict[str, Any] = field(default_factory=dict)
    expect: ExpectConfig = field(default_factory=ExpectConfig)


@dataclass
class SuiteConfig:
    """Full configuration for a test suite loaded from a YAML file."""

    name: str
    model: str
    api_provider: str
    tests: List[TestCase]
    prompt: Optional[str] = None
    prompt_file: Optional[str] = None
    baseline_model: Optional[str] = None
    source_path: Optional[Path] = None

    def get_system_prompt(self) -> Optional[str]:
        """Return the resolved system prompt, loading from file if needed."""
        if self.prompt:
            return self.prompt
        if self.prompt_file:
            prompt_path = Path(self.prompt_file)
            if not prompt_path.is_absolute() and self.source_path:
                prompt_path = self.source_path.parent / prompt_path
            if not prompt_path.exists():
                raise FileNotFoundError(
                    f"Prompt file not found: {prompt_path}\n"
                    f"Check the 'prompt_file' field in your suite YAML."
                )
            return prompt_path.read_text(encoding="utf-8").strip()
        return None


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _resolve_env_vars(value: Any) -> Any:
    """
    Recursively resolve ${ENV_VAR} placeholders in YAML values.

    Raises:
        EnvironmentError: If a referenced variable is not set.
    """
    if isinstance(value, str):
        def replacer(match: re.Match) -> str:
            var_name = match.group(1)
            resolved = os.environ.get(var_name)
            if resolved is None:
                raise EnvironmentError(
                    f"Environment variable '{var_name}' is required but not set.\n"
                    f"Set it with: export {var_name}=<your-value>"
                )
            return resolved
        return _ENV_VAR_PATTERN.sub(replacer, value)
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


def _parse_expect(raw: Optional[Dict[str, Any]]) -> ExpectConfig:
    """Parse the 'expect' block from a raw YAML dict."""
    if not raw:
        return ExpectConfig()
    return ExpectConfig(
        format=raw.get("format"),
        max_words=raw.get("max_words"),
        min_words=raw.get("min_words"),
        max_bullets=raw.get("max_bullets"),
        min_bullets=raw.get("min_bullets"),
        max_sentences=raw.get("max_sentences"),
        required_phrases=raw.get("required_phrases") or [],
        forbidden_phrases=raw.get("forbidden_phrases") or [],
        forbidden_patterns=raw.get("forbidden_patterns") or [],
        required_patterns=raw.get("required_patterns") or [],
        semantic_threshold=float(raw.get("semantic_threshold", 0.75)),
        tone=raw.get("tone"),
    )


def _parse_test_case(raw: Dict[str, Any]) -> TestCase:
    """Parse a single test case from a raw YAML dict."""
    required = ("name", "input")
    for field_name in required:
        if field_name not in raw:
            raise ValueError(
                f"Test case is missing required field '{field_name}'.\n"
                f"Each test must have at least 'name' and 'input'."
            )
    return TestCase(
        name=raw["name"],
        input=raw["input"],
        variables=raw.get("variables") or {},
        expect=_parse_expect(raw.get("expect")),
    )


def load_suite(path: Path) -> SuiteConfig:
    """
    Load and validate a single YAML test suite file.

    Args:
        path: Absolute or relative path to the .yaml suite file.

    Returns:
        A fully validated SuiteConfig instance.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If required fields are missing or invalid.
        yaml.YAMLError: If the file is not valid YAML.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Suite file not found: {path}\n"
            f"Check the path passed to --suite."
        )

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"Suite file must be a YAML mapping (dict), got: {type(raw).__name__}")

    raw = _resolve_env_vars(raw)

    for required_field in ("name", "model", "tests"):
        if required_field not in raw:
            raise ValueError(
                f"Suite file '{path.name}' is missing required field '{required_field}'.\n"
                f"Required fields: name, model, tests"
            )

    if not isinstance(raw["tests"], list) or len(raw["tests"]) == 0:
        raise ValueError(
            f"Suite '{raw['name']}' must contain at least one test in the 'tests' list."
        )

    api_provider = raw.get("api_provider", "groq")
    valid_providers = {"groq", "openai", "anthropic"}
    if api_provider not in valid_providers:
        raise ValueError(
            f"Unknown api_provider '{api_provider}'. "
            f"Valid options: {', '.join(sorted(valid_providers))}"
        )

    tests = [_parse_test_case(t) for t in raw["tests"]]

    return SuiteConfig(
        name=raw["name"],
        model=raw["model"],
        api_provider=api_provider,
        tests=tests,
        prompt=raw.get("prompt"),
        prompt_file=raw.get("prompt_file"),
        baseline_model=raw.get("baseline_model"),
        source_path=path,
    )


def load_suites_from_directory(directory: Path) -> List[SuiteConfig]:
    """
    Discover and load all .yaml suite files in a directory.

    Args:
        directory: Path to search for YAML suite files.

    Returns:
        List of SuiteConfig instances, sorted by filename.
    """
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    yaml_files = sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.yml"))
    if not yaml_files:
        raise ValueError(f"No YAML suite files found in: {directory}")

    suites = []
    errors = []
    for yaml_file in yaml_files:
        try:
            suites.append(load_suite(yaml_file))
        except Exception as exc:
            errors.append(f"  {yaml_file.name}: {exc}")

    if errors:
        error_list = "\n".join(errors)
        raise ValueError(f"Errors loading suites from {directory}:\n{error_list}")

    return suites


def get_api_key(provider: str) -> str:
    """
    Retrieve the API key for a given provider from environment variables.

    Args:
        provider: One of 'groq', 'openai', 'anthropic'.

    Returns:
        The API key string.

    Raises:
        EnvironmentError: If the required environment variable is not set.
    """
    env_map = {
        "groq": "GROQ_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    env_var = env_map.get(provider)
    if env_var is None:
        raise ValueError(f"Unknown provider '{provider}'. Valid: {', '.join(env_map)}")

    key = os.environ.get(env_var)
    if not key:
        raise EnvironmentError(
            f"API key for '{provider}' not found.\n"
            f"Set it with: export {env_var}=<your-api-key>\n"
            f"Get a free Groq key at: https://console.groq.com"
        )
    return key
