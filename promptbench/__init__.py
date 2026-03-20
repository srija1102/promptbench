"""
promptbench — A prompt regression testing framework for LLMs.

Catch prompt regressions before they reach production by defining expected
behaviors in YAML files and automatically testing against any LLM.
"""

__version__ = "0.1.0"
__author__ = "promptbench contributors"

from promptbench.runner import TestResult, TestRunner
from promptbench.config import SuiteConfig, TestCase

__all__ = ["TestResult", "TestRunner", "SuiteConfig", "TestCase", "__version__"]
