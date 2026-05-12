# Repository Guidelines

## Project Structure & Module Organization

Python package for multi-agent portfolio diagnosis.

- `src/mafin/` contains the application code.
- `src/mafin/agents/` defines LLM agent classes, including specialists and the moderator.
- `src/mafin/graph/` defines LangGraph state and graph construction.
- `src/mafin/data/` defines portfolio and user profile schemas.
- `src/mafin/llm/` contains model client integrations, currently Ollama.
- `scripts/smoke.py` runs a smoke test and supports dry-run route inspection.
- `tests/` contains unit tests for schemas, workload conversion, graph compilation, tracing, and baseline helpers.

## Build, Test, and Development Commands

Use `uv`.

```bash
uv sync
```

Installs dependencies into the local virtual environment.

```bash
uv run python scripts/smoke.py
```

Runs the LangGraph smoke flow. Requires Ollama and the configured model.

```bash
uv run python scripts/smoke.py --dry-run --route-preset b4r
```

Compiles the graph and prints the Ollama route without invoking LLMs.

```bash
uv run python scripts/benchmark_models.py --route-preset b4r
```

Checks structured-output compatibility for the unique models in a route.

```bash
uv run python scripts/run_baselines.py --snapshot data/snapshots/example.json --dry-run
```

Prints the baseline execution plan without invoking LLMs.

```bash
uv run python scripts/run_baselines.py --workload data/workloads/sample.json --limit-cases 2 --dry-run
```

Checks the workload execution plan before running multiple cases.

```bash
uv run python -B -c "from mafin.graph.build import build_graph; print(type(build_graph()).__name__)"
```

Checks graph compilation without invoking an LLM.

```bash
uv run ruff check .
```

## Coding Style & Naming Conventions

Target Python is 3.11+. Use 4-space indentation and type hints. Keep modules small and role-specific.

Naming conventions:

- Agent classes use `PascalCase`, for example `TechnicalAgent`.
- Module names use lowercase snake case, for example `portfolio.py`.
- Graph state fields use concise snake case, for example `market_data` and `debate_rounds`.

Ruff is configured in `pyproject.toml` with line length `100` and rules `E`, `F`, `I`, `UP`, `B`, `SIM`.

## Testing Guidelines

The project uses `pytest`.

Recommended conventions:

- Place tests in `tests/`.
- Name files `test_<module>.py`.
- Name test functions `test_<behavior>()`.
- Mock LLM calls; unit tests should not require Ollama or network access.

Run `uv run pytest` before changing prompts, routing, tracing, workload loading, or baseline logic. Use the smoke script as an integration check, not as a test substitute.

## Agent Execution Model

Local development uses workstation Ollama to validate structure and prompts. Real experiments should use Ollama Cloud models. Use `DEFAULT_LLM` for B4-H and role-specific variables such as `TECHNICAL_LLM`, `BULL_LLM`, and `MODERATOR_LLM` for B4-R. If `OLLAMA_API_KEY` is set and `OLLAMA_HOST` is omitted, the code uses direct Ollama Cloud at `https://ollama.com` and strips `:cloud` from model names. Set `OLLAMA_HOST=http://localhost:11434` to force the local daemon. Keep concurrency controlled by `LLM_MAX_CONCURRENCY`; reduce it to `1` when the provider returns overload errors. Inference latency is secondary to reproducibility, comparable outputs, and API compatibility.

Structured output defaults to `STRUCTURED_OUTPUT_MODE=manual`, which sends the Pydantic JSON schema inside the prompt and parses the returned JSON locally. Use `STRUCTURED_OUTPUT_MODE=json_schema` only when the target Ollama endpoint reliably supports native structured output. `OLLAMA_NUM_PREDICT` caps response length and `OLLAMA_TIMEOUT_SECONDS` controls HTTP timeout for long-running calls.

Set `LLM_MAX_CONCURRENCY=3` for Ollama Cloud Pro. This parallelizes independent specialist agents and B2 self-consistency samples while keeping debate, moderation, cases, and baselines sequential. Do not set a value above the provider limit.

## Commit & Pull Request Guidelines

History only shows an initial commit, so no convention is established. Use short imperative messages, such as `Add risk agent skeleton` or `Fix Ollama model route`.

Pull requests need:

- Concise summary.
- Any setup or model requirements, especially Ollama model tags.
- Validation commands.
- Notes on evaluation impact if changing prompts, routing, baselines, or metrics.

## Security & Configuration Tips

Do not commit `.env`, logs, cache files, or local datasets. Keep API keys outside the repository and prefer environment variables. Ollama Cloud authentication uses `OLLAMA_API_KEY`; the smoke dry run only reports whether it is configured and never prints the value.
