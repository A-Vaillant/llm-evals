"""Tests for PromptTemplate message formatting."""

import pytest
from llm_evals.config import PromptDefinition
from llm_evals.prompt import PromptTemplate


class TestPromptTemplate:
    def test_user_only_formats_row(self):
        defn = PromptDefinition(user="Answer: {question}")
        tmpl = PromptTemplate(defn)
        msgs = tmpl.format({"question": "What is 2+2?"})
        assert msgs == [{"role": "user", "content": "Answer: What is 2+2?"}]

    def test_system_and_user(self):
        defn = PromptDefinition(system="You are helpful.", user="Q: {question}")
        tmpl = PromptTemplate(defn)
        msgs = tmpl.format({"question": "Sky color?"})
        assert msgs[0] == {"role": "system", "content": "You are helpful."}
        assert msgs[1] == {"role": "user", "content": "Q: Sky color?"}

    def test_prefill_appended_as_assistant(self):
        defn = PromptDefinition(user="{question}")
        tmpl = PromptTemplate(defn, prefill="<answer>")
        msgs = tmpl.format({"question": "Which?"})
        assert msgs[-1] == {"role": "assistant", "content": "<answer>"}

    def test_missing_field_raises(self):
        defn = PromptDefinition(user="{question} {choices}")
        tmpl = PromptTemplate(defn)
        with pytest.raises(KeyError):
            tmpl.format({"question": "Q?"})  # missing choices

    def test_extra_fields_ignored(self):
        defn = PromptDefinition(user="{question}")
        tmpl = PromptTemplate(defn)
        msgs = tmpl.format({"question": "Q?", "expected": "A", "category": "test"})
        assert len(msgs) == 1

    def test_multiline_user_template(self):
        defn = PromptDefinition(user="{question}\n\nA) {a}\nB) {b}")
        tmpl = PromptTemplate(defn)
        row = {"question": "Color?", "a": "Red", "b": "Blue"}
        msgs = tmpl.format(row)
        assert "A) Red" in msgs[0]["content"]
        assert "B) Blue" in msgs[0]["content"]
