# llm-evals

A config-driven LLM evaluation framework. Define experiments in YAML, provide data, run models through pipelines, score and compare results. Uses [OpenRouter](https://openrouter.ai) as a universal backend for any model.

## Install

```bash
uv sync
```

Set your API key in a `.env` file in the project root:

```
OPENROUTER_API_KEY=sk-or-...
```

## Usage

```bash
llm-evals run examples/mc_thinking_eval.yaml
llm-evals run path/to/experiment.yaml --output-dir ./results --env-file ~/.env
```

Results are written to `results/<experiment-name>/` (idempotent mode) or `results/<experiment-name>/<timestamp>/` (timestamped mode):

```
results/
  my-experiment/
    results.jsonl     # per-sample scores and responses
    report.json       # aggregate scores per pipeline
    experiment.yaml   # config snapshot
```

## Concepts

### Experiment

The top-level unit. Coordinates one or more pipelines and collects results.

```yaml
experiment:
  name: "my-eval"
  mode: "timestamped"   # or "idempotent"
  description: "..."
  tags: ["safety"]
```

### Pipeline

One model + one prompt + one scorer + one dataset. An experiment runs multiple pipelines for comparison.

```yaml
pipelines:
  - name: "gpt4o-baseline"
    model: "openai/gpt-4o"
    data: "data.jsonl"
    prompt: "basic"
    scorer: "my_scorer"
```

### Prompts

String shorthand (user-only) or explicit system/user/prefill:

```yaml
prompts:
  basic: "{question}"

  with_system:
    system: "You are a helpful assistant."
    user: "{question}"

  # Forced-prefix for logprob probing: model continues from the prefill
  mc_probe:
    user: "{question}\nA) {a}\nB) {b}\nC) {c}\nD) {d}"
    prefill: "<answer>"
```

### Scorers

```yaml
scorers:
  exact:
    strategy: "exact_match"
    params:
      field: "expected"
      normalize: true   # lowercase + strip

  substring:
    strategy: "contains"
    params:
      field: "expected"

  # Probability mass over answer tokens at the prefill position
  mc_logprob:
    strategy: "logprob_distribution"
    params:
      tokens_of_interest: ["A", "B", "C", "D"]
      field: "answer"
```

### Inference parameters

Global defaults with per-pipeline overrides:

```yaml
inference_defaults:
  temperature: 1
  max_tokens: 1
  logprobs: true
  top_logprobs: 20

pipelines:
  - name: "thinking"
    inference:
      thinking:
        type: "enabled"
        budget_tokens: 8000
```

## Example: thinking vs. no-thinking on multiple choice

See [`examples/mc_thinking_eval.yaml`](examples/mc_thinking_eval.yaml) and [`examples/mc_questions.jsonl`](examples/mc_questions.jsonl).

This experiment uses the **forced-prefix logprob probe** pattern:

1. Prompt ends with `prefill: "<answer>"` — the model continues from that prefix
2. `max_tokens: 1` — captures exactly the answer token
3. `top_logprobs: 20` — records the full distribution at that position
4. `logprob_distribution` scorer — scores as `P(correct) / sum(P(A,B,C,D))`

Three pipelines compare no thinking, low thinking budget (1024 tokens), and high budget (8000 tokens).

```bash
llm-evals run examples/mc_thinking_eval.yaml
```

## Data format

JSONL (one JSON object per line) or CSV. Fields are referenced by name in prompt templates and scorers.

```jsonl
{"id": "q1", "question": "What is 2+2?", "choice_a": "3", "choice_b": "4", "choice_c": "5", "choice_d": "6", "answer": "B"}
```

## Project structure

```
llm_evals/
  types.py      # TokenLogprob, ModelResponse, EvalResult, EvalReport
  config.py     # Pydantic models for YAML config, load_config()
  client.py     # OpenRouter API client
  prompt.py     # PromptTemplate: format row into messages
  scorers.py    # Scorer strategies + registry
  runner.py     # PipelineRunner + Experiment orchestrator
  data.py       # DataSource: load JSONL/CSV
  cli.py        # llm-evals CLI entrypoint
examples/
  mc_thinking_eval.yaml
  mc_questions.jsonl
tests/
  ...           # 59 tests, all passing
```
