"""OpenRouter API client.

Wraps the openai SDK with OpenRouter's base URL. Key capabilities:
- Forced assistant prefill: pass an assistant message as the last message
  in the messages list; the model continues from that prefix.
- Logprob extraction: parses top_logprobs from the API response into
  the framework's TokenLogprob types.
- Thinking passthrough: any InferenceParams.thinking dict is forwarded
  as-is to the API (model-specific; e.g. Anthropic extended thinking).

Note on prefill support: OpenRouter forwards prefill to providers that
support it (Anthropic). OpenAI-hosted models do not support prefill via
this mechanism — the assistant message will be ignored or cause an error.
Target Anthropic models for the forced-prefix logprob probe pattern.
"""

from __future__ import annotations

from typing import Any

import openai

from llm_evals.config import InferenceParams
from llm_evals.types import ModelResponse, TokenLogprob

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient:
    """Thin client over the OpenAI SDK pointed at OpenRouter.

    Args:
        api_key: OpenRouter API key.
        base_url: Override the base URL (useful for testing local proxies).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = OPENROUTER_BASE_URL,
    ) -> None:
        self._openai = openai.OpenAI(api_key=api_key, base_url=base_url)

    def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        params: InferenceParams,
    ) -> ModelResponse:
        """Make a chat completion request and return a ModelResponse.

        Args:
            messages: OpenAI-style message list. If the last message has
                role ``"assistant"``, it acts as a forced prefix (prefill).
            model: OpenRouter model ID (e.g. ``"anthropic/claude-3-7-sonnet"``).
            params: Inference parameters to apply.

        Returns:
            ModelResponse with text and optional logprob data.
        """
        kwargs: dict[str, Any] = {"model": model, "messages": messages}

        if params.temperature is not None:
            kwargs["temperature"] = params.temperature
        if params.max_tokens is not None:
            kwargs["max_tokens"] = params.max_tokens
        if params.logprobs:
            kwargs["logprobs"] = True
        if params.top_logprobs is not None:
            kwargs["top_logprobs"] = params.top_logprobs
        if params.thinking is not None:
            kwargs["thinking"] = params.thinking

        completion = self._openai.chat.completions.create(**kwargs)
        choice = completion.choices[0]

        logprobs: list[TokenLogprob] | None = None
        top_logprobs: list[list[TokenLogprob]] | None = None

        if choice.logprobs is not None and choice.logprobs.content:
            logprobs = []
            top_logprobs = []
            for i, token_lp in enumerate(choice.logprobs.content):
                logprobs.append(TokenLogprob(token=token_lp.token, logprob=token_lp.logprob, rank=0))
                if token_lp.top_logprobs:
                    top_logprobs.append([
                        TokenLogprob(token=alt.token, logprob=alt.logprob, rank=rank)
                        for rank, alt in enumerate(token_lp.top_logprobs)
                    ])
                else:
                    top_logprobs.append([])

        return ModelResponse(
            text=choice.message.content or "",
            model=model,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )
