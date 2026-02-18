"""Tests for scorer strategies."""

import math
import pytest
from llm_evals.scorers import build_scorer
from llm_evals.types import ModelResponse, TokenLogprob


def _response(text: str, top_logprobs: list[list[TokenLogprob]] | None = None) -> ModelResponse:
    logprobs = None
    if top_logprobs:
        logprobs = [lps[0] for lps in top_logprobs if lps]
    return ModelResponse(
        text=text, model="test/model", logprobs=logprobs, top_logprobs=top_logprobs
    )


class TestExactMatch:
    def test_match(self):
        scorer = build_scorer("exact_match", {"field": "expected"})
        assert scorer.score(_response("A"), {"expected": "A"}) == 1.0

    def test_no_match(self):
        scorer = build_scorer("exact_match", {"field": "expected"})
        assert scorer.score(_response("B"), {"expected": "A"}) == 0.0

    def test_normalize(self):
        scorer = build_scorer("exact_match", {"field": "expected", "normalize": True})
        assert scorer.score(_response("  a  "), {"expected": "A"}) == 1.0

    def test_no_normalize_case_sensitive(self):
        scorer = build_scorer("exact_match", {"field": "expected", "normalize": False})
        assert scorer.score(_response("a"), {"expected": "A"}) == 0.0


class TestContains:
    def test_substring_found(self):
        scorer = build_scorer("contains", {"field": "expected"})
        assert scorer.score(_response("The answer is A."), {"expected": "A"}) == 1.0

    def test_substring_not_found(self):
        scorer = build_scorer("contains", {"field": "expected"})
        assert scorer.score(_response("The answer is B."), {"expected": "A"}) == 0.0


class TestLogprobDistribution:
    def _make_top_lps(self, token_probs: dict[str, float]) -> list[list[TokenLogprob]]:
        """Build a single-position top_logprobs from {token: prob} dict."""
        entries = [
            TokenLogprob(token=tok, logprob=math.log(p), rank=i)
            for i, (tok, p) in enumerate(token_probs.items())
        ]
        return [entries]

    def test_correct_answer_highest_prob(self):
        # A has 70% prob, B has 20%, C has 10%
        top_lps = self._make_top_lps({"A": 0.7, "B": 0.2, "C": 0.1})
        resp = _response("A", top_logprobs=top_lps)
        scorer = build_scorer(
            "logprob_distribution",
            {"tokens_of_interest": ["A", "B", "C", "D"], "field": "answer"},
        )
        result = scorer.score(resp, {"answer": "A"})
        # Score = prob mass on the correct answer / total mass on tokens_of_interest
        assert result == pytest.approx(0.7 / (0.7 + 0.2 + 0.1))

    def test_wrong_answer_low_score(self):
        top_lps = self._make_top_lps({"A": 0.7, "B": 0.2, "C": 0.1})
        resp = _response("A", top_logprobs=top_lps)
        scorer = build_scorer(
            "logprob_distribution",
            {"tokens_of_interest": ["A", "B", "C", "D"], "field": "answer"},
        )
        result = scorer.score(resp, {"answer": "B"})
        assert result == pytest.approx(0.2 / (0.7 + 0.2 + 0.1))

    def test_missing_logprobs_raises(self):
        resp = _response("A")  # no logprobs
        scorer = build_scorer(
            "logprob_distribution",
            {"tokens_of_interest": ["A", "B", "C", "D"], "field": "answer"},
        )
        with pytest.raises(ValueError, match="logprob"):
            scorer.score(resp, {"answer": "A"})

    def test_none_of_interest_in_distribution(self):
        # Top logprobs only contain "X" and "Y", not A/B/C/D
        top_lps = self._make_top_lps({"X": 0.9, "Y": 0.1})
        resp = _response("X", top_logprobs=top_lps)
        scorer = build_scorer(
            "logprob_distribution",
            {"tokens_of_interest": ["A", "B", "C", "D"], "field": "answer"},
        )
        # No relevant tokens found: score is 0
        result = scorer.score(resp, {"answer": "A"})
        assert result == 0.0
