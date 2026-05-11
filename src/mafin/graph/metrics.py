from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


def _empty_runtime_metrics() -> dict[str, Any]:
    return {"llm_calls": 0, "calls_by_model": {}}


def instrument_node(
    name: str,
    runner: Any,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Wrap a LangGraph node with latency and LLM-call accounting."""

    def wrapped(state: dict[str, Any]) -> dict[str, Any]:
        reset = getattr(runner, "reset_runtime_metrics", None)
        if callable(reset):
            reset()

        started = time.perf_counter()
        update = runner.run(state)
        elapsed = time.perf_counter() - started

        consume = getattr(runner, "consume_runtime_metrics", None)
        runtime_metrics = consume() if callable(consume) else _empty_runtime_metrics()
        node_metric = {
            "node": name,
            "elapsed_seconds": round(elapsed, 6),
            "llm_calls": runtime_metrics["llm_calls"],
            "calls_by_model": runtime_metrics["calls_by_model"],
            "cost_proxy_unit": "llm_call",
            "cost_proxy_value": runtime_metrics["llm_calls"],
        }

        existing_metrics = update.get("execution_metrics", [])
        return {
            **update,
            "execution_metrics": [*existing_metrics, node_metric],
        }

    return wrapped


def summarize_execution(state: dict[str, Any]) -> dict[str, Any]:
    metrics = state.get("execution_metrics", [])
    calls_by_model: dict[str, int] = {}
    total_latency = 0.0
    total_calls = 0

    for metric in metrics:
        total_latency += float(metric.get("elapsed_seconds", 0.0))
        total_calls += int(metric.get("llm_calls", 0))
        for model, calls in metric.get("calls_by_model", {}).items():
            calls_by_model[model] = calls_by_model.get(model, 0) + int(calls)

    debate_status = state.get("debate_status", {})
    return {
        "execution_summary": {
            "total_node_latency_seconds": round(total_latency, 6),
            "total_llm_calls": total_calls,
            "calls_by_model": calls_by_model,
            "debate_rounds_completed": debate_status.get("rounds_completed", 0),
            "debate_stopped_by": debate_status.get("stopped_by"),
            "cost_proxy_unit": "llm_call",
            "cost_proxy_value": total_calls,
            "cost_note": (
                "Proxy operacional. O custo real no Ollama Cloud deve ser calculado "
                "com telemetria de uso/GPU quando disponível."
            ),
        }
    }
