"""YAML config parsing and validation for llm-evals.

The config schema mirrors the DESIGN.md DSL exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

# ──────────────────────────────────────────────
# Allowed scorer strategy names (validated at parse time)
# ──────────────────────────────────────────────
VALID_STRATEGIES = {
    "exact_match",
    "contains",
    "llm_judge",
    "logprob_distribution",
    "custom",
}


# ──────────────────────────────────────────────
# Sub-models
# ──────────────────────────────────────────────


class ExperimentMeta(BaseModel):
    """Top-level experiment metadata block."""

    model_config = ConfigDict(frozen=True)

    name: str
    mode: Literal["idempotent", "timestamped"] = "idempotent"
    description: str = ""
    tags: list[str] = []
    metadata: dict[str, Any] = {}


class PromptDefinition(BaseModel):
    """A resolved prompt with optional system message and assistant prefill."""

    model_config = ConfigDict(frozen=True)

    user: str
    system: str | None = None
    # When set, an assistant message with this content is appended as the last
    # message, implementing forced-prefix / prefill for logprob probing.
    prefill: str | None = None


class InferenceParams(BaseModel):
    """Inference parameters for a model call.

    All fields are optional; unset fields are omitted from the API call.
    Per-pipeline params override these defaults.
    """

    model_config = ConfigDict(frozen=True)

    temperature: float = 0
    max_tokens: int | None = None
    logprobs: bool = False
    top_logprobs: int | None = None
    # Thinking / extended reasoning (model-specific; passed through as-is)
    thinking: dict[str, Any] | None = None

    def merge(self, override: InferenceParams) -> InferenceParams:
        """Return a new InferenceParams with override values applied."""
        base = self.model_dump(exclude_unset=False)
        over = override.model_dump(exclude_unset=True)
        return InferenceParams(**{**base, **over})


class ScorerConfig(BaseModel):
    """Scorer strategy + params."""

    model_config = ConfigDict(frozen=True)

    strategy: str
    params: dict[str, Any] = {}

    @field_validator("strategy")
    @classmethod
    def strategy_must_be_valid(cls, v: str) -> str:
        if v not in VALID_STRATEGIES:
            raise ValueError(
                f"Unknown scorer strategy {v!r}. Valid: {sorted(VALID_STRATEGIES)}"
            )
        return v


class PipelineConfig(BaseModel):
    """Single pipeline definition."""

    model_config = ConfigDict(frozen=True)

    name: str
    model: str
    data: str
    prompt: str
    # scorer can be a named reference (str) or an inline ScorerConfig dict
    scorer: str | ScorerConfig
    inference: InferenceParams = InferenceParams()


# ──────────────────────────────────────────────
# Top-level config
# ──────────────────────────────────────────────


class ExperimentConfig(BaseModel):
    """Full parsed experiment config."""

    model_config = ConfigDict(frozen=True)

    experiment: ExperimentMeta
    # Raw prompts: str shorthand or {system, user} dict
    prompts: dict[str, str | PromptDefinition]
    inference_defaults: InferenceParams = InferenceParams()
    # Named scorers (optional; pipelines may also define scorers inline)
    scorers: dict[str, ScorerConfig] = {}
    pipelines: list[PipelineConfig]

    @model_validator(mode="before")
    @classmethod
    def normalise_prompts(cls, data: Any) -> Any:
        """Coerce prompt string shorthands to PromptDefinition dicts."""
        raw_prompts = data.get("prompts", {})
        normalised: dict[str, Any] = {}
        for key, val in raw_prompts.items():
            if isinstance(val, str):
                normalised[key] = PromptDefinition(user=val)
            else:
                normalised[key] = val
        data["prompts"] = normalised
        return data

    def resolved_prompt(self, name: str) -> PromptDefinition:
        """Return the PromptDefinition for *name*, raising KeyError if absent."""
        prompt = self.prompts[name]
        if isinstance(prompt, PromptDefinition):
            return prompt
        # Already normalised in model_validator; this branch shouldn't be hit.
        return PromptDefinition(user=prompt)  # type: ignore[arg-type]

    def resolved_scorer(self, pipeline: PipelineConfig) -> ScorerConfig:
        """Resolve a pipeline's scorer — inline definition or named reference."""
        scorer = pipeline.scorer
        if isinstance(scorer, ScorerConfig):
            return scorer
        # It's a name reference
        if scorer not in self.scorers:
            raise KeyError(f"Scorer {scorer!r} not defined in top-level scorers block")
        return self.scorers[scorer]

    def resolved_inference(self, pipeline: PipelineConfig) -> InferenceParams:
        """Merge inference_defaults with pipeline-level overrides."""
        return self.inference_defaults.merge(pipeline.inference)

    @classmethod
    def from_yaml(cls, text: str) -> ExperimentConfig:
        """Parse YAML string into a validated ExperimentConfig."""
        raw = yaml.safe_load(text)
        return cls.model_validate(raw)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment config from a YAML file."""
    text = Path(path).read_text(encoding="utf-8")
    return ExperimentConfig.from_yaml(text)
