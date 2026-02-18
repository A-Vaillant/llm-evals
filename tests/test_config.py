"""Tests for YAML config / DSL parsing."""
import pytest
from llm_evals.config import ExperimentConfig, load_config


MINIMAL_CONFIG = """
experiment:
  name: "test-eval"

prompts:
  basic: "{prompt}"

pipelines:
  - name: "gpt4o-basic"
    model: "openai/gpt-4o"
    data: "data.jsonl"
    prompt: "basic"
    scorer:
      strategy: "exact_match"
      params:
        field: "expected"
"""

FULL_CONFIG = """
experiment:
  name: "safety-refusal"
  mode: "timestamped"
  description: "Test model safety refusal behavior"
  tags: ["safety", "refusal"]
  metadata:
    author: "test-user"

prompts:
  safety_test:
    system: "You are a helpful assistant."
    user: "{prompt}"

inference_defaults:
  temperature: 0
  max_tokens: 1
  logprobs: true
  top_logprobs: 5

scorers:
  refusal_check:
    strategy: "exact_match"
    params:
      field: "expected"
      normalize: true

pipelines:
  - name: "gpt4o-safety"
    model: "openai/gpt-4o"
    data: "safety_data.jsonl"
    prompt: "safety_test"
    scorer: "refusal_check"

  - name: "claude-safety"
    model: "anthropic/claude-3.5-sonnet"
    data: "safety_data.jsonl"
    prompt: "safety_test"
    scorer: "refusal_check"
"""

COT_COMPARISON_CONFIG = """
experiment:
  name: "cot-comparison"
  mode: "idempotent"

prompts:
  direct: "{question}\\nAnswer with just the letter."
  with_cot: "{question}\\nThink step by step, then answer."

inference_defaults:
  temperature: 0
  logprobs: true
  top_logprobs: 5

scorers:
  mc_match:
    strategy: "exact_match"
    params:
      field: "answer"
      normalize: true

pipelines:
  - name: "gpt4o-direct"
    model: "openai/gpt-4o"
    data: "mc_questions.jsonl"
    prompt: "direct"
    scorer: "mc_match"

  - name: "gpt4o-cot"
    model: "openai/gpt-4o"
    data: "mc_questions.jsonl"
    prompt: "with_cot"
    scorer: "mc_match"
"""


class TestExperimentConfig:
    def test_minimal(self):
        cfg = ExperimentConfig.from_yaml(MINIMAL_CONFIG)
        assert cfg.experiment.name == "test-eval"
        assert cfg.experiment.mode == "idempotent"  # default
        assert len(cfg.pipelines) == 1
        assert cfg.pipelines[0].name == "gpt4o-basic"
        assert cfg.pipelines[0].model == "openai/gpt-4o"

    def test_prompts_string_shorthand(self):
        cfg = ExperimentConfig.from_yaml(MINIMAL_CONFIG)
        assert "basic" in cfg.prompts
        # String shorthand becomes user-only prompt
        p = cfg.resolved_prompt("basic")
        assert p.user == "{prompt}"
        assert p.system is None

    def test_prompts_with_system(self):
        cfg = ExperimentConfig.from_yaml(FULL_CONFIG)
        p = cfg.resolved_prompt("safety_test")
        assert p.system == "You are a helpful assistant."
        assert p.user == "{prompt}"

    def test_full_experiment_metadata(self):
        cfg = ExperimentConfig.from_yaml(FULL_CONFIG)
        assert cfg.experiment.mode == "timestamped"
        assert cfg.experiment.description == "Test model safety refusal behavior"
        assert "safety" in cfg.experiment.tags
        assert cfg.experiment.metadata["author"] == "test-user"

    def test_inference_defaults(self):
        cfg = ExperimentConfig.from_yaml(FULL_CONFIG)
        assert cfg.inference_defaults.temperature == 0
        assert cfg.inference_defaults.max_tokens == 1
        assert cfg.inference_defaults.logprobs is True
        assert cfg.inference_defaults.top_logprobs == 5

    def test_inference_defaults_optional(self):
        cfg = ExperimentConfig.from_yaml(MINIMAL_CONFIG)
        assert cfg.inference_defaults.temperature == 0
        assert cfg.inference_defaults.logprobs is False

    def test_named_scorers(self):
        cfg = ExperimentConfig.from_yaml(FULL_CONFIG)
        assert "refusal_check" in cfg.scorers
        assert cfg.scorers["refusal_check"].strategy == "exact_match"
        assert cfg.scorers["refusal_check"].params["field"] == "expected"

    def test_pipeline_inline_scorer(self):
        cfg = ExperimentConfig.from_yaml(MINIMAL_CONFIG)
        scorer = cfg.resolved_scorer(cfg.pipelines[0])
        assert scorer.strategy == "exact_match"
        assert scorer.params["field"] == "expected"

    def test_pipeline_ref_scorer(self):
        cfg = ExperimentConfig.from_yaml(FULL_CONFIG)
        scorer = cfg.resolved_scorer(cfg.pipelines[0])
        assert scorer.strategy == "exact_match"
        assert scorer.params["normalize"] is True

    def test_multi_pipeline_different_prompts(self):
        cfg = ExperimentConfig.from_yaml(COT_COMPARISON_CONFIG)
        assert len(cfg.pipelines) == 2
        assert cfg.pipelines[0].prompt == "direct"
        assert cfg.pipelines[1].prompt == "with_cot"

    def test_pipeline_per_pipeline_inference_override(self):
        yaml = """
experiment:
  name: "test"

prompts:
  basic: "{prompt}"

inference_defaults:
  temperature: 0
  max_tokens: 256

pipelines:
  - name: "p1"
    model: "openai/gpt-4o"
    data: "data.jsonl"
    prompt: "basic"
    inference:
      max_tokens: 1
      logprobs: true
    scorer:
      strategy: "exact_match"
      params:
        field: "expected"
"""
        cfg = ExperimentConfig.from_yaml(yaml)
        resolved = cfg.resolved_inference(cfg.pipelines[0])
        assert resolved.max_tokens == 1  # overridden
        assert resolved.temperature == 0  # inherited from defaults
        assert resolved.logprobs is True  # overridden

    def test_load_from_file(self, tmp_path):
        config_file = tmp_path / "eval.yaml"
        config_file.write_text(MINIMAL_CONFIG)
        cfg = load_config(config_file)
        assert cfg.experiment.name == "test-eval"

    def test_invalid_scorer_strategy_rejected(self):
        bad = MINIMAL_CONFIG.replace("exact_match", "nonexistent_strategy")
        with pytest.raises(Exception):
            ExperimentConfig.from_yaml(bad)

    def test_pipeline_references_missing_prompt(self):
        bad = MINIMAL_CONFIG.replace('"basic"', '"nonexistent"').replace(
            "basic:", "real_prompt:"
        )
        cfg = ExperimentConfig.from_yaml(bad)
        with pytest.raises(KeyError):
            cfg.resolved_prompt("nonexistent")

    def test_experiment_mode_validation(self):
        bad = MINIMAL_CONFIG.replace("test-eval", "test-eval\"\n  mode: \"invalid_mode")
        with pytest.raises(Exception):
            ExperimentConfig.from_yaml(bad)
