"""Scorer strategies for evaluating model responses.

Each scorer takes a ModelResponse and the original data row and returns
a float score in [0.0, 1.0].

Usage:
    scorer = build_scorer("exact_match", {"field": "expected", "normalize": True})
    score = scorer.score(response, row)

New strategies are added by subclassing BaseScorer and registering with
@register_scorer("strategy_name").
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

from llm_evals.types import ModelResponse

# ──────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────

_REGISTRY: dict[str, type[BaseScorer]] = {}


def register_scorer(name: str):
    """Class decorator to register a scorer strategy by name."""
    def decorator(cls: type[BaseScorer]) -> type[BaseScorer]:
        _REGISTRY[name] = cls
        return cls
    return decorator


def build_scorer(strategy: str, params: dict[str, Any]) -> BaseScorer:
    """Instantiate a scorer by strategy name and params dict.

    Args:
        strategy: Registered strategy name.
        params: Strategy-specific parameters.

    Returns:
        Configured BaseScorer instance.

    Raises:
        KeyError: If strategy name is not registered.
    """
    if strategy not in _REGISTRY:
        raise KeyError(f"Unknown scorer strategy {strategy!r}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[strategy](params)


# ──────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────


class BaseScorer(ABC):
    """Abstract base for all scorer strategies."""

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params

    @abstractmethod
    def score(self, response: ModelResponse, row: dict[str, Any]) -> float:
        """Score a model response against the original data row.

        Returns:
            Float in [0.0, 1.0].
        """


# ──────────────────────────────────────────────
# Built-in strategies
# ──────────────────────────────────────────────


@register_scorer("exact_match")
class ExactMatchScorer(BaseScorer):
    """1.0 if response text exactly matches the expected field, else 0.0.

    Params:
        field (str): Row field name to compare against.
        normalize (bool): Lowercase + strip whitespace before comparing. Default False.
    """

    def score(self, response: ModelResponse, row: dict[str, Any]) -> float:
        expected: str = str(row[self.params["field"]])
        actual: str = response.text
        if self.params.get("normalize", False):
            expected = expected.strip().lower()
            actual = actual.strip().lower()
        return 1.0 if actual == expected else 0.0


@register_scorer("contains")
class ContainsScorer(BaseScorer):
    """1.0 if expected field value is a substring of the response text, else 0.0.

    Params:
        field (str): Row field name whose value to search for.
    """

    def score(self, response: ModelResponse, row: dict[str, Any]) -> float:
        expected: str = str(row[self.params["field"]])
        return 1.0 if expected in response.text else 0.0


@register_scorer("logprob_distribution")
class LogprobDistributionScorer(BaseScorer):
    """Score based on the model's probability mass over answer-choice tokens.

    Designed for the forced-prefix logprob probe pattern: the model sees
    ``<answer>`` as an assistant prefill and the first generated token is the
    answer letter. This scorer extracts the probability distribution over the
    tokens_of_interest at that position, then returns the normalized mass on
    the correct answer.

    Score = P(correct_token) / sum(P(t) for t in tokens_of_interest)

    If none of the tokens_of_interest appear in the top_logprobs distribution,
    the score is 0.0.

    Params:
        tokens_of_interest (list[str]): Answer tokens to track (e.g. ["A","B","C","D"]).
        field (str): Row field containing the correct answer token.
        position (int): Which output token position to read logprobs from. Default 0.
    """

    def score(self, response: ModelResponse, row: dict[str, Any]) -> float:
        if response.top_logprobs is None:
            raise ValueError(
                "logprob_distribution scorer requires top_logprobs in the ModelResponse. "
                "Set logprobs=true and top_logprobs=N in inference params."
            )

        tokens_of_interest: list[str] = self.params["tokens_of_interest"]
        correct: str = str(row[self.params["field"]])
        position: int = self.params.get("position", 0)

        if position >= len(response.top_logprobs):
            raise ValueError(
                f"Requested logprob position {position} but response only has "
                f"{len(response.top_logprobs)} token positions."
            )

        top_lps = response.top_logprobs[position]
        # Build token → linear prob map from the distribution at this position
        prob_map: dict[str, float] = {lp.token: math.exp(lp.logprob) for lp in top_lps}

        total = sum(prob_map.get(t, 0.0) for t in tokens_of_interest)
        if total == 0.0:
            return 0.0

        correct_mass = prob_map.get(correct, 0.0)
        return correct_mass / total


@register_scorer("llm_judge")
class LLMJudgeScorer(BaseScorer):
    """Placeholder — LLM-as-judge scoring (not yet implemented).

    Params:
        judge_model (str): OpenRouter model ID for the judge.
        rubric (str): Scoring rubric prompt.
        score_map (dict): Maps judge response strings to float scores.
    """

    def score(self, response: ModelResponse, row: dict[str, Any]) -> float:
        raise NotImplementedError("llm_judge scorer is not yet implemented")


@register_scorer("custom")
class CustomScorer(BaseScorer):
    """Escape hatch: load and call a user-defined Python scoring function.

    Params:
        module (str): Dotted module path to import.
        function (str): Function name within that module.
    """

    def score(self, response: ModelResponse, row: dict[str, Any]) -> float:
        import importlib
        mod = importlib.import_module(self.params["module"])
        fn = getattr(mod, self.params["function"])
        return float(fn(response, row))
