"""Pipeline runner and experiment orchestrator.

PipelineRunner: executes one pipeline end-to-end (load data → format prompts
    → call model → score results).

Experiment: coordinates multiple pipelines, writes results to disk.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_evals.client import OpenRouterClient
from llm_evals.config import ExperimentConfig, PipelineConfig
from llm_evals.data import DataSource
from llm_evals.prompt import PromptTemplate
from llm_evals.scorers import build_scorer
from llm_evals.types import EvalReport, EvalResult


class PipelineRunner:
    """Runs a single pipeline: loads data, calls the model, scores results.

    Args:
        config: The full experiment config (used for prompt/scorer/inference resolution).
        client: Configured OpenRouterClient (or compatible mock).
    """

    def __init__(self, config: ExperimentConfig, client: OpenRouterClient) -> None:
        self._config = config
        self._client = client

    def run(self, pipeline: PipelineConfig) -> list[EvalResult]:
        """Execute the pipeline and return per-row EvalResults.

        Args:
            pipeline: The pipeline to run.

        Returns:
            List of EvalResult, one per data row.
        """
        rows = DataSource.load(pipeline.data)
        prompt_defn = self._config.resolved_prompt(pipeline.prompt)
        template = PromptTemplate(prompt_defn)
        scorer_cfg = self._config.resolved_scorer(pipeline)
        scorer = build_scorer(scorer_cfg.strategy, scorer_cfg.params)
        inference = self._config.resolved_inference(pipeline)

        results: list[EvalResult] = []
        for row in rows:
            messages = template.format(row)
            response = self._client.complete(
                messages=messages,
                model=pipeline.model,
                params=inference,
            )
            score = scorer.score(response, row)
            results.append(
                EvalResult(
                    sample_id=row["_id"],
                    pipeline=pipeline.name,
                    model=pipeline.model,
                    response=response,
                    row=row,
                    score=score,
                )
            )

        return results


class Experiment:
    """Orchestrates multiple pipelines and persists results to disk.

    Args:
        config: Full experiment config.
        client: Configured OpenRouterClient (or compatible mock).
        output_dir: Root directory for result output. Defaults to ``./results``.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        client: OpenRouterClient,
        output_dir: str | Path = "results",
    ) -> None:
        self._config = config
        self._client = client
        self._output_dir = Path(output_dir)

    def run(self) -> EvalReport:
        """Run all pipelines and write results to disk.

        Returns:
            EvalReport aggregating all pipeline results.
        """
        runner = PipelineRunner(config=self._config, client=self._client)
        all_results: list[EvalResult] = []

        for pipeline in self._config.pipelines:
            all_results.extend(runner.run(pipeline))

        report = EvalReport(results=all_results)
        self._write(report)
        return report

    def _output_path(self) -> Path:
        """Resolve the output directory for this experiment run."""
        meta = self._config.experiment
        base = self._output_dir / meta.name
        if meta.mode == "timestamped":
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
            return base / stamp
        return base  # idempotent: just the name, overwritten each run

    def _write(self, report: EvalReport) -> None:
        out = self._output_path()
        out.mkdir(parents=True, exist_ok=True)

        # Per-sample results
        results_file = out / "results.jsonl"
        with results_file.open("w", encoding="utf-8") as fh:
            for result in report.results:
                fh.write(result.model_dump_json() + "\n")

        # Aggregate report
        report_file = out / "report.json"
        report_file.write_text(
            json.dumps(report.summary(), indent=2),
            encoding="utf-8",
        )

        # Config snapshot
        config_file = out / "experiment.yaml"
        config_file.write_text(
            json.dumps(self._config.model_dump(), indent=2),
            encoding="utf-8",
        )
