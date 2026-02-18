"""Tests for PipelineRunner and Experiment orchestration."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from llm_evals.config import ExperimentConfig
from llm_evals.runner import PipelineRunner, Experiment
from llm_evals.types import EvalResult, ModelResponse, TokenLogprob


SIMPLE_CONFIG = """
experiment:
  name: "test-run"
  mode: "idempotent"

prompts:
  basic: "{{question}}"

pipelines:
  - name: "test-pipeline"
    model: "openai/gpt-4o"
    data: "{data_path}"
    prompt: "basic"
    scorer:
      strategy: "exact_match"
      params:
        field: "answer"
"""


def _mock_client(text: str = "A") -> MagicMock:
    client = MagicMock()
    client.complete.return_value = ModelResponse(text=text, model="openai/gpt-4o")
    return client


class TestPipelineRunner:
    def test_runs_all_rows(self, tmp_path):
        data = tmp_path / "data.jsonl"
        data.write_text(
            json.dumps({"question": "Q1?", "answer": "A"}) + "\n"
            + json.dumps({"question": "Q2?", "answer": "B"}) + "\n"
        )
        cfg = ExperimentConfig.from_yaml(SIMPLE_CONFIG.format(data_path=str(data)))
        pipeline = cfg.pipelines[0]

        client = _mock_client("A")
        runner = PipelineRunner(config=cfg, client=client)
        results = runner.run(pipeline)

        assert len(results) == 2
        assert client.complete.call_count == 2

    def test_scores_correctly(self, tmp_path):
        data = tmp_path / "data.jsonl"
        data.write_text(
            json.dumps({"question": "Q1?", "answer": "A"}) + "\n"
            + json.dumps({"question": "Q2?", "answer": "B"}) + "\n"
        )
        cfg = ExperimentConfig.from_yaml(SIMPLE_CONFIG.format(data_path=str(data)))
        pipeline = cfg.pipelines[0]

        # Client always returns "A"; Q1 matches, Q2 does not
        client = _mock_client("A")
        runner = PipelineRunner(config=cfg, client=client)
        results = runner.run(pipeline)

        scores = [r.score for r in results]
        assert scores[0] == 1.0
        assert scores[1] == 0.0

    def test_result_fields_populated(self, tmp_path):
        data = tmp_path / "data.jsonl"
        data.write_text(json.dumps({"question": "Q1?", "answer": "A"}) + "\n")
        cfg = ExperimentConfig.from_yaml(SIMPLE_CONFIG.format(data_path=str(data)))
        pipeline = cfg.pipelines[0]

        runner = PipelineRunner(config=cfg, client=_mock_client("A"))
        results = runner.run(pipeline)
        r = results[0]

        assert isinstance(r, EvalResult)
        assert r.pipeline == "test-pipeline"
        assert r.model == "openai/gpt-4o"
        assert r.row["question"] == "Q1?"
        assert r.sample_id == "0"

    def test_messages_include_prefill_when_configured(self, tmp_path):
        """Prefill in prompt config is forwarded to the client as assistant message."""
        data = tmp_path / "data.jsonl"
        data.write_text(json.dumps({"question": "Q1?", "answer": "A"}) + "\n")

        yaml = """
experiment:
  name: "prefill-test"

prompts:
  probe:
    user: "{{question}}"
    prefill: "<answer>"

pipelines:
  - name: "p1"
    model: "openai/gpt-4o"
    data: "{data_path}"
    prompt: "probe"
    scorer:
      strategy: "exact_match"
      params:
        field: "answer"
""".format(data_path=str(data))

        cfg = ExperimentConfig.from_yaml(yaml)
        pipeline = cfg.pipelines[0]
        client = _mock_client("A")
        runner = PipelineRunner(config=cfg, client=client)
        runner.run(pipeline)

        call_messages = client.complete.call_args.kwargs["messages"]
        assert call_messages[-1] == {"role": "assistant", "content": "<answer>"}


class TestExperiment:
    def test_writes_results_jsonl(self, tmp_path):
        data = tmp_path / "data.jsonl"
        data.write_text(json.dumps({"question": "Q?", "answer": "A"}) + "\n")
        cfg = ExperimentConfig.from_yaml(SIMPLE_CONFIG.format(data_path=str(data)))

        client = _mock_client("A")
        exp = Experiment(config=cfg, client=client, output_dir=tmp_path / "results")
        exp.run()

        results_file = tmp_path / "results" / "test-run" / "results.jsonl"
        assert results_file.exists()
        lines = [json.loads(l) for l in results_file.read_text().strip().splitlines()]
        assert len(lines) == 1
        assert lines[0]["pipeline"] == "test-pipeline"

    def test_writes_report_json(self, tmp_path):
        data = tmp_path / "data.jsonl"
        data.write_text(json.dumps({"question": "Q?", "answer": "A"}) + "\n")
        cfg = ExperimentConfig.from_yaml(SIMPLE_CONFIG.format(data_path=str(data)))

        client = _mock_client("A")
        exp = Experiment(config=cfg, client=client, output_dir=tmp_path / "results")
        report = exp.run()

        report_file = tmp_path / "results" / "test-run" / "report.json"
        assert report_file.exists()
        data_json = json.loads(report_file.read_text())
        assert "test-pipeline" in data_json

    def test_timestamped_mode_creates_subdirectory(self, tmp_path):
        yaml = """
experiment:
  name: "ts-test"
  mode: "timestamped"

prompts:
  basic: "{{question}}"

pipelines:
  - name: "p1"
    model: "openai/gpt-4o"
    data: "{data_path}"
    prompt: "basic"
    scorer:
      strategy: "exact_match"
      params:
        field: "answer"
""".format(data_path=str(tmp_path / "data.jsonl"))
        (tmp_path / "data.jsonl").write_text(json.dumps({"question": "Q?", "answer": "A"}) + "\n")

        cfg = ExperimentConfig.from_yaml(yaml)
        exp = Experiment(config=cfg, client=_mock_client("A"), output_dir=tmp_path / "results")
        exp.run()

        # Should be results/ts-test/<timestamp>/results.jsonl
        ts_dirs = list((tmp_path / "results" / "ts-test").iterdir())
        assert len(ts_dirs) == 1
        assert (ts_dirs[0] / "results.jsonl").exists()
