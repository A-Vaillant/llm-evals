"""Tests for DataSource JSONL/CSV loading."""

import json
import pytest
from llm_evals.data import DataSource


class TestDataSource:
    def test_load_jsonl(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text(
            json.dumps({"prompt": "Q1", "expected": "A"}) + "\n"
            + json.dumps({"prompt": "Q2", "expected": "B"}) + "\n"
        )
        rows = DataSource.load(f)
        assert len(rows) == 2
        assert rows[0]["prompt"] == "Q1"
        assert rows[1]["expected"] == "B"

    def test_load_jsonl_skips_blank_lines(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text(
            json.dumps({"a": 1}) + "\n"
            + "\n"
            + json.dumps({"a": 2}) + "\n"
        )
        rows = DataSource.load(f)
        assert len(rows) == 2

    def test_load_csv(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("prompt,expected\nQ1,A\nQ2,B\n")
        rows = DataSource.load(f)
        assert len(rows) == 2
        assert rows[0]["prompt"] == "Q1"
        assert rows[1]["expected"] == "B"

    def test_unsupported_extension_raises(self, tmp_path):
        f = tmp_path / "data.tsv"
        f.write_text("a\tb\n1\t2\n")
        with pytest.raises(ValueError, match="Unsupported"):
            DataSource.load(f)

    def test_sample_ids_assigned(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text(
            json.dumps({"prompt": "Q1"}) + "\n"
            + json.dumps({"prompt": "Q2"}) + "\n"
        )
        rows = DataSource.load(f)
        # Each row gets a stable _id (index-based if no id field present)
        assert "_id" in rows[0]
        assert rows[0]["_id"] != rows[1]["_id"]

    def test_existing_id_field_preserved(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text(
            json.dumps({"id": "custom-1", "prompt": "Q1"}) + "\n"
        )
        rows = DataSource.load(f)
        assert rows[0]["_id"] == "custom-1"
