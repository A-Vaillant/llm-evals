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
            pipeline="gpt4o-baseline",
            model="openai/gpt-4o",
            response=r,
            row={"prompt": "test", "expected": "A"},
            score=1.0,
        )
        assert result.score == 1.0
        assert result.sample_id == "q1"
        assert result.pipeline == "gpt4o-baseline"

    def test_row_preserved(self):
        r = ModelResponse(text="A", model="openai/gpt-4o")
        row = {"prompt": "test", "expected": "A", "category": "safety"}
        result = EvalResult(
            sample_id="q1",
            pipeline="p1",
            model="openai/gpt-4o",
            response=r,
            row=row,
            score=1.0,
        )
        assert result.row["category"] == "safety"

    def test_optional_metadata(self):
        r = ModelResponse(text="A", model="openai/gpt-4o")
        result = EvalResult(
            sample_id="q1",
            pipeline="p1",
            model="openai/gpt-4o",
            response=r,
            row={"prompt": "test"},
            score=1.0,
            metadata={"latency_ms": 230},
        )
        assert result.metadata["latency_ms"] == 230


class TestEvalReport:
    def _make_result(self, pipeline, model, score):
        return EvalResult(
            sample_id=f"q{score}",
            pipeline=pipeline,
            model=model,
            response=ModelResponse(text="x", model=model),
            row={"prompt": "test"},
            score=score,
        )

    def test_mean_score_by_pipeline(self):
        results = [
            self._make_result("p1", "model-a", 1.0),
            self._make_result("p1", "model-a", 0.0),
        ]
        report = EvalReport(results=results)
        assert report.mean_score(pipeline="p1") == 0.5

    def test_multi_pipeline(self):
        results = [
            self._make_result("gpt4o-direct", "openai/gpt-4o", 1.0),
            self._make_result("gpt4o-cot", "openai/gpt-4o", 0.5),
            self._make_result("claude-direct", "anthropic/claude-3.5-sonnet", 0.8),
        ]
        report = EvalReport(results=results)
        assert report.mean_score(pipeline="gpt4o-direct") == 1.0
        assert report.mean_score(pipeline="gpt4o-cot") == 0.5
        assert set(report.pipelines) == {"gpt4o-direct", "gpt4o-cot", "claude-direct"}

    def test_summary(self):
        results = [
            self._make_result("p1", "model-a", 1.0),
            self._make_result("p1", "model-a", 0.0),
            self._make_result("p2", "model-b", 0.75),
        ]
        report = EvalReport(results=results)
        table = report.summary()
        assert table["p1"]["mean_score"] == 0.5
        assert table["p1"]["n"] == 2
        assert table["p1"]["model"] == "model-a"
        assert table["p2"]["mean_score"] == 0.75
        assert table["p2"]["n"] == 1
