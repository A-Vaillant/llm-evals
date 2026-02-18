"""CLI entrypoint for llm-evals.

Usage:
    llm-evals run path/to/experiment.yaml
    llm-evals run path/to/experiment.yaml --output-dir ./results
    llm-evals run path/to/experiment.yaml --env-file .env

The OPENROUTER_API_KEY environment variable must be set (or present in the
loaded .env file).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _cmd_run(args: argparse.Namespace) -> None:
    # Dotenv loading happens here, at the boundary, before anything else.
    from dotenv import load_dotenv

    env_file = Path(args.env_file) if args.env_file else Path(".env")
    if env_file.exists():
        load_dotenv(env_file)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    from llm_evals.client import OpenRouterClient
    from llm_evals.config import load_config
    from llm_evals.runner import Experiment

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(config_path)
    client = OpenRouterClient(api_key=api_key)
    output_dir = Path(args.output_dir)

    print(f"Experiment : {cfg.experiment.name}")
    print(f"Mode       : {cfg.experiment.mode}")
    print(f"Pipelines  : {', '.join(p.name for p in cfg.pipelines)}")
    print(f"Output     : {output_dir}")
    print()

    for i, pipeline in enumerate(cfg.pipelines, 1):
        print(f"[{i}/{len(cfg.pipelines)}] Running pipeline: {pipeline.name} ({pipeline.model})")

    exp = Experiment(config=cfg, client=client, output_dir=output_dir)
    report = exp.run()

    print()
    print("Results:")
    summary = report.summary()
    col_w = max(len(k) for k in summary) + 2
    print(f"  {'Pipeline':<{col_w}}  {'Score':>6}  {'N':>4}  Model")
    print(f"  {'-'*col_w}  {'------':>6}  {'----':>4}  -----")
    for pipeline, stats in summary.items():
        print(
            f"  {pipeline:<{col_w}}  {stats['mean_score']:>6.3f}"
            f"  {stats['n']:>4}  {stats['model']}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-evals",
        description="Config-driven LLM evaluation framework",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run an experiment from a YAML config")
    run_p.add_argument("config", help="Path to experiment YAML file")
    run_p.add_argument(
        "--output-dir",
        default="results",
        metavar="DIR",
        help="Root directory for results (default: ./results)",
    )
    run_p.add_argument(
        "--env-file",
        default=None,
        metavar="FILE",
        help="Path to .env file (default: .env in cwd)",
    )
    run_p.set_defaults(func=_cmd_run)

    return parser


def main() -> None:
    """CLI entrypoint registered in pyproject.toml."""
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)
