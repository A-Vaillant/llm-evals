"""DataSource: load evaluation data from JSONL or CSV files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class DataSource:
    """Namespace for data loading utilities."""

    @staticmethod
    def load(path: str | Path) -> list[dict[str, Any]]:
        """Load rows from a JSONL or CSV file.

        Each row is a plain dict. A ``_id`` key is injected:
        - If the row already has an ``id`` field, ``_id`` mirrors it.
        - Otherwise ``_id`` is the 0-based row index (as a string).

        Args:
            path: Path to a ``.jsonl`` or ``.csv`` file.

        Returns:
            List of row dicts, each with a ``_id`` key.

        Raises:
            ValueError: For unsupported file extensions.
            FileNotFoundError: If the path does not exist.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            rows = DataSource._load_jsonl(path)
        elif suffix == ".csv":
            rows = DataSource._load_csv(path)
        else:
            raise ValueError(
                f"Unsupported data format {suffix!r}. Supported: .jsonl, .csv"
            )

        for i, row in enumerate(rows):
            row["_id"] = str(row["id"]) if "id" in row else str(i)

        return rows

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    @staticmethod
    def _load_csv(path: Path) -> list[dict[str, Any]]:
        with path.open(newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
