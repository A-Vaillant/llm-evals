"""Tests for OpenRouter client.

All tests use a mocked openai.OpenAI to avoid real API calls.
"""

from unittest.mock import MagicMock, patch
import pytest
from llm_evals.client import OpenRouterClient
from llm_evals.config import InferenceParams
from llm_evals.types import ModelResponse, TokenLogprob


def _make_mock_completion(
    text: str = "A",
    model: str = "openai/gpt-4o",
    logprobs_data: list | None = None,
):
    """Build a minimal mock openai ChatCompletion object."""
    choice = MagicMock()
    choice.message.content = text

    if logprobs_data is not None:
        token_entries = []
        for entry in logprobs_data:
            token_lp = MagicMock()
            token_lp.token = entry["token"]
            token_lp.logprob = entry["logprob"]
            top = []
            for i, alt in enumerate(entry.get("top_logprobs", [])):
                alt_mock = MagicMock()
                alt_mock.token = alt["token"]
                alt_mock.logprob = alt["logprob"]
                top.append(alt_mock)
            token_lp.top_logprobs = top
            token_entries.append(token_lp)
        choice.logprobs = MagicMock()
        choice.logprobs.content = token_entries
    else:
        choice.logprobs = None

    completion = MagicMock()
    completion.choices = [choice]
    completion.model = model
    return completion


class TestOpenRouterClient:
    def _client(self):
        return OpenRouterClient(api_key="test-key")

    def test_returns_model_response(self):
        mock_completion = _make_mock_completion("Hello")
        with patch("llm_evals.client.openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = mock_completion
            client = self._client()
            resp = client.complete(
                messages=[{"role": "user", "content": "Hi"}],
                model="openai/gpt-4o",
                params=InferenceParams(),
            )
        assert isinstance(resp, ModelResponse)
        assert resp.text == "Hello"

    def test_logprobs_parsed(self):
        lp_data = [
            {
                "token": "A",
                "logprob": -0.3,
                "top_logprobs": [
                    {"token": "A", "logprob": -0.3},
                    {"token": "B", "logprob": -1.5},
                ],
            }
        ]
        mock_completion = _make_mock_completion("A", logprobs_data=lp_data)
        with patch("llm_evals.client.openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = mock_completion
            client = self._client()
            resp = client.complete(
                messages=[{"role": "user", "content": "Q?"}, {"role": "assistant", "content": "<answer>"}],
                model="openai/gpt-4o",
                params=InferenceParams(logprobs=True, top_logprobs=5, max_tokens=1),
            )
        assert resp.logprobs is not None
        assert len(resp.logprobs) == 1
        assert resp.logprobs[0].token == "A"
        assert resp.logprobs[0].logprob == pytest.approx(-0.3)
        assert resp.top_logprobs is not None
        assert len(resp.top_logprobs[0]) == 2
        assert resp.top_logprobs[0][0].token == "A"
        assert resp.top_logprobs[0][1].token == "B"

    def test_thinking_params_passed_through(self):
        mock_completion = _make_mock_completion("A")
        with patch("llm_evals.client.openai.OpenAI") as MockOpenAI:
            create_fn = MockOpenAI.return_value.chat.completions.create
            create_fn.return_value = mock_completion
            client = self._client()
            client.complete(
                messages=[{"role": "user", "content": "Q?"}],
                model="anthropic/claude-3-7-sonnet",
                params=InferenceParams(
                    thinking={"type": "enabled", "budget_tokens": 8000}
                ),
            )
            call_kwargs = create_fn.call_args.kwargs
            assert call_kwargs.get("thinking") == {"type": "enabled", "budget_tokens": 8000}

    def test_no_logprobs_when_not_requested(self):
        mock_completion = _make_mock_completion("Hello")
        with patch("llm_evals.client.openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = mock_completion
            client = self._client()
            resp = client.complete(
                messages=[{"role": "user", "content": "Hi"}],
                model="openai/gpt-4o",
                params=InferenceParams(),
            )
        assert resp.logprobs is None
        assert resp.top_logprobs is None
