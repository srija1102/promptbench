"""
semantic_check.py — Semantic similarity evaluation using sentence-transformers.

Uses the 'all-MiniLM-L6-v2' model (runs locally, no API key required) to
compute cosine similarity between the current output and the stored baseline.
Fails if similarity drops below the configured threshold (default 0.75).

The model is loaded lazily and cached as a module-level singleton to avoid
reloading on every test run.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Module-level singleton so the model is loaded only once per process
_model = None
_MODEL_NAME = "all-MiniLM-L6-v2"


def _get_model():
    """
    Lazily load and cache the SentenceTransformer model.

    Returns:
        A loaded SentenceTransformer instance.

    Raises:
        ImportError: If sentence-transformers is not installed.
    """
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for semantic similarity checks.\n"
                "Install it with: pip install sentence-transformers"
            ) from exc
        logger.debug("Loading SentenceTransformer model '%s'...", _MODEL_NAME)
        _model = SentenceTransformer(_MODEL_NAME)
        logger.debug("Model loaded.")
    return _model


def _cosine_similarity(vec_a, vec_b) -> float:
    """
    Compute cosine similarity between two numpy arrays.

    Args:
        vec_a: First embedding vector.
        vec_b: Second embedding vector.

    Returns:
        Similarity score in range [-1, 1]; practically [0, 1] for sentence embeddings.
    """
    import numpy as np  # type: ignore
    dot = float(np.dot(vec_a, vec_b))
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticChecker:
    """
    Evaluator that computes cosine similarity between current and baseline outputs
    using a locally-running sentence-transformers model.
    """

    DEFAULT_THRESHOLD = 0.75

    def check(
        self,
        current_output: str,
        baseline_output: str,
        threshold: Optional[float] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Compare *current_output* against *baseline_output* using semantic similarity.

        Args:
            current_output: The output produced in the current test run.
            baseline_output: The previously stored baseline output.
            threshold: Minimum cosine similarity required to pass (default 0.75).

        Returns:
            (passed, failure_reason).
        """
        if not current_output.strip() or not baseline_output.strip():
            return False, "Cannot compute semantic similarity: one or both strings are empty."

        effective_threshold = threshold if threshold is not None else self.DEFAULT_THRESHOLD

        try:
            similarity = self.similarity(current_output, baseline_output)
        except Exception as exc:
            return False, f"Semantic similarity computation failed: {exc}"

        if similarity < effective_threshold:
            return (
                False,
                f"Semantic similarity {similarity:.3f} is below threshold {effective_threshold:.3f}. "
                f"The output may have drifted significantly from the baseline.",
            )
        return True, None

    @staticmethod
    def similarity(text_a: str, text_b: str) -> float:
        """
        Compute the cosine similarity between two strings.

        Args:
            text_a: First string.
            text_b: Second string.

        Returns:
            Cosine similarity in range [0, 1].
        """
        model = _get_model()
        embeddings = model.encode([text_a, text_b], convert_to_numpy=True)
        return _cosine_similarity(embeddings[0], embeddings[1])
