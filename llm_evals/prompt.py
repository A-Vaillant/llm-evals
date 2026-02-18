"""PromptTemplate: format data rows into OpenAI-style message lists."""

from __future__ import annotations

from typing import Any

from llm_evals.config import PromptDefinition


class PromptTemplate:
    """Formats a data row into a messages list for an LLM API call.

    Args:
        definition: The prompt definition (user template + optional system).
        prefill: Optional assistant prefill string. When set, an assistant
            message is appended as the final message, implementing the
            forced-prefix pattern for logprob probing.
    """

    def __init__(self, definition: PromptDefinition, prefill: str | None = None) -> None:
        self._definition = definition
        # Explicit prefill arg takes precedence over definition.prefill
        self._prefill = prefill if prefill is not None else definition.prefill

    def format(self, row: dict[str, Any]) -> list[dict[str, str]]:
        """Format *row* fields into a messages list.

        Args:
            row: Data row dict. All ``{field}`` placeholders in the template
                must be present as keys.

        Returns:
            List of ``{"role": ..., "content": ...}`` dicts.

        Raises:
            KeyError: If a template placeholder references a missing row field.
        """
        messages: list[dict[str, str]] = []

        if self._definition.system is not None:
            messages.append({
                "role": "system",
                "content": self._definition.system.format_map(row),
            })

        messages.append({
            "role": "user",
            "content": self._definition.user.format_map(row),
        })

        if self._prefill is not None:
            messages.append({"role": "assistant", "content": self._prefill})

        return messages
