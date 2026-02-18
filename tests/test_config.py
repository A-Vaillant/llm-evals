"""Tests for YAML config / DSL parsing."""
import pytest
import tempfile
from pathlib import Path
from llm_evals.config import EvalConfig, load_config


MINIMAL_CONFIG = """
eval:
  name: "test-eval"
  models:
    - "openai/gpt-4o"

data:
  path: "data.jsonl"
  input_field: "prompt"
  label_field: "expected"

inference:
  mode: "generate"

scoring:
  method: "exact_match"
"""

FULL_CONFIG = """
eval:
  name: "safety-refusal"
  description: "Test model safety refusal behavior"
  models:
    - "openai/gpt-4o"
    - "anthropic/claude-3.5-sonnet"

data:
  path: "safety_data.jsonl"
  input_field: "prompt"
  label_field: "expected"

inference:
  mode: "multiple_choice"
  choices: ["safe", "unsafe"]
  system_prompt: "You are a helpful assistant."
  logprobs: true
  top_logprobs: 5
  temperature: 0
  max_tokens: 1

scoring:
  method: "exact_match"
"""

LOGPROB_CONFIG = """
eval:
  name: "token-distribution"
  models:
    - "openai/gpt-4o"

data:
  path: "test_data.jsonl"
  input_field: "prompt"
  label_field: "answer"

inference:
  mode: "logprobs_only"
  logprobs: true
  top_logprobs: 10
  temperature: 0
  max_tokens: 1

scoring:
  method: "logprob_distribution"
"""


class TestLoadConfig:
    def test_minimal(self):
        cfg = EvalConfig.from_yaml(MINIMAL_CONFIG)
        assert cfg.eval.name == "test-eval"
        assert cfg.eval.models == ["openai/gpt-4o"]
        assert cfg.data.input_field == "prompt"
        assert cfg.inference.mode == "generate"
        assert cfg.scoring.method == "exact_match"

    def test_full(self):
        cfg = EvalConfig.from_yaml(FULL_CONFIG)
        assert len(cfg.eval.models) == 2
        assert cfg.inference.mode == "multiple_choice"
        assert cfg.inference.choices == ["safe", "unsafe"]
        assert cfg.inference.logprobs is True
        assert cfg.inference.top_logprobs == 5
        assert cfg.inference.temperature == 0
        assert cfg.inference.max_tokens == 1
        assert cfg.inference.system_prompt == "You are a helpful assistant."

    def test_logprob_mode(self):
        cfg = EvalConfig.from_yaml(LOGPROB_CONFIG)
        assert cfg.inference.mode == "logprobs_only"
        assert cfg.inference.top_logprobs == 10
        assert cfg.scoring.method == "logprob_distribution"

    def test_load_from_file(self, tmp_path):
        config_file = tmp_path / "eval.yaml"
        config_file.write_text(MINIMAL_CONFIG)
        cfg = load_config(config_file)
        assert cfg.eval.name == "test-eval"

    def test_defaults(self):
        cfg = EvalConfig.from_yaml(MINIMAL_CONFIG)
        assert cfg.inference.logprobs is False
        assert cfg.inference.temperature == 0
        assert cfg.inference.max_tokens is None
        assert cfg.inference.choices is None

    def test_invalid_mode_rejected(self):
        bad = MINIMAL_CONFIG.replace("generate", "invalid_mode")
        with pytest.raises(Exception):
            EvalConfig.from_yaml(bad)

    def test_invalid_scoring_rejected(self):
        bad = MINIMAL_CONFIG.replace("exact_match", "nonexistent_method")
        with pytest.raises(Exception):
            EvalConfig.from_yaml(bad)
