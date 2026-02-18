"""Core data types for llm-evals.

All types are immutable Pydantic models.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, computed_field


class TokenLogprob(BaseModel):
    """Log-probability for a single token at a single position."""

    model_config = ConfigDict(frozen=True)

    token: str
    logprob: float
    rank: int  # 0 = the chosen token, 1..N = alternatives

    @computed_field  # type: ignore[prop-decorator]
    @property
    def prob(self) -> float:
        """Linear probability derived from logprob."""
        return math.exp(self.logprob)


class ModelResponse(BaseModel):
    """Raw output from a single model call."""

    model_config = ConfigDict(frozen=True)

    text: str
    model: str
    # Per-position chosen-token logprobs (one entry per output token).
    logprobs: list[TokenLogprob] | None = None
    # Per-position top-N alternatives (outer list = positions, inner = alternatives).
    top_logprobs: list[list[TokenLogprob]] | None = None


class EvalResult(BaseModel):
    """Scored result for a single data row through a single pipeline."""

    model_config = ConfigDict(frozen=True)

    sample_id: str
    pipeline: str
    model: str
    response: ModelResponse
    row: dict[str, Any]
    score: float
    metadata: dict[str, Any] = {}


class EvalReport(BaseModel):
    """Aggregate results across all pipelines in an experiment."""

    model_config = ConfigDict(frozen=True)

    results: list[EvalResult]

    @property
    def pipelines(self) -> list[str]:
        """Unique pipeline names present in results, in insertion order."""
        seen: dict[str, None] = {}
        for r in self.results:
            seen[r.pipeline] = None
        return list(seen)

    def mean_score(self, *, pipeline: str) -> float:
        """Mean score for a given pipeline."""
        scores = [r.score for r in self.results if r.pipeline == pipeline]
        if not scores:
            raise KeyError(f"No results for pipeline {pipeline!r}")
        return sum(scores) / len(scores)

    def summary(self) -> dict[str, dict[str, Any]]:
        """Per-pipeline summary: mean_score, n, model."""
        out: dict[str, dict[str, Any]] = {}
        for pipeline in self.pipelines:
            rows = [r for r in self.results if r.pipeline == pipeline]
            out[pipeline] = {
                "mean_score": sum(r.score for r in rows) / len(rows),
                "n": len(rows),
                "model": rows[0].model,
            }
        return out
