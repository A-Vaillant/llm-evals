"""Tests for core data types."""
import pytest
from llm_evals.types import (
    TokenLogprob,
    ModelResponse,
    EvalResult,
    EvalReport,
)


class TestTokenLogprob:
    def test_create(self):
        t = TokenLogprob(token="A", logprob=-0.5, rank=0)
        assert t.token == "A"
        assert t.logprob == -0.5
        assert t.rank == 0

    def test_prob_property(self):
        import math

        t = TokenLogprob(token="A", logprob=-1.0, rank=0)
        assert abs(t.prob - math.exp(-1.0)) < 1e-9


class TestModelResponse:
    def test_minimal(self):
        r = ModelResponse(text="Hello", model="openai/gpt-4o")
        assert r.text == "Hello"
        assert r.logprobs is None

    def test_with_logprobs(self):
        lp = TokenLogprob(token="A", logprob=-0.1, rank=0)
        r = ModelResponse(text="A", model="openai/gpt-4o", logprobs=[lp])
        assert len(r.logprobs) == 1

    def test_with_top_logprobs(self):
        top = [
            TokenLogprob(token="A", logprob=-0.1, rank=0),
            TokenLogprob(token="B", logprob=-2.3, rank=1),
        ]
        r = ModelResponse(
            text="A", model="openai/gpt-4o", top_logprobs=[top]
        )
        assert len(r.top_logprobs) == 1
        assert len(r.top_logprobs[0]) == 2


class TestEvalResult:
    def test_create(self):
        r = ModelResponse(text="A", model="openai/gpt-4o")
        result = EvalResult(
            sample_id="q1",
            model="openai/gpt-4o",
            response=r,
            expected="A",
            score=1.0,
        )
        assert result.score == 1.0
        assert result.sample_id == "q1"

    def test_optional_metadata(self):
        r = ModelResponse(text="A", model="openai/gpt-4o")
        result = EvalResult(
            sample_id="q1",
            model="openai/gpt-4o",
            response=r,
            expected="A",
            score=1.0,
            metadata={"latency_ms": 230},
        )
        assert result.metadata["latency_ms"] == 230


class TestEvalReport:
    def test_from_results(self):
        r1 = EvalResult(
            sample_id="q1",
            model="openai/gpt-4o",
            response=ModelResponse(text="A", model="openai/gpt-4o"),
            expected="A",
            score=1.0,
        )
        r2 = EvalResult(
            sample_id="q2",
            model="openai/gpt-4o",
            response=ModelResponse(text="B", model="openai/gpt-4o"),
            expected="A",
            score=0.0,
        )
        report = EvalReport(results=[r1, r2])
        assert report.mean_score("openai/gpt-4o") == 0.5

    def test_multi_model(self):
        results = [
            EvalResult(
                sample_id="q1",
                model="model-a",
                response=ModelResponse(text="A", model="model-a"),
                expected="A",
                score=1.0,
            ),
            EvalResult(
                sample_id="q1",
                model="model-b",
                response=ModelResponse(text="B", model="model-b"),
                expected="A",
                score=0.0,
            ),
        ]
        report = EvalReport(results=results)
        assert report.mean_score("model-a") == 1.0
        assert report.mean_score("model-b") == 0.0
        assert set(report.models) == {"model-a", "model-b"}

    def test_summary_table(self):
        results = [
            EvalResult(
                sample_id="q1",
                model="model-a",
                response=ModelResponse(text="A", model="model-a"),
                expected="A",
                score=1.0,
            ),
        ]
        report = EvalReport(results=results)
        table = report.summary()
        assert "model-a" in table
        assert isinstance(table, dict)
        assert table["model-a"]["mean_score"] == 1.0
        assert table["model-a"]["n"] == 1
