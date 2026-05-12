from __future__ import annotations

import copy
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from contextvars import copy_context
from typing import Any, Literal

from mafin.baselines.single import (
    SingleShotDiagnosisAgent,
    aggregate_self_consistency,
)
from mafin.config import DEFAULT_LLM, LLM_MAX_CONCURRENCY, TRACE_DB_PATH, get_model_route
from mafin.graph.build import build_graph
from mafin.tracing import SQLiteTraceStore, trace_run
from mafin.tracing.store import stable_hash

BaselineName = Literal["b1", "b2", "b3", "b4h", "b4r"]
BASELINE_NAMES: tuple[BaselineName, ...] = ("b1", "b2", "b3", "b4h", "b4r")


def _prepare_state(state: dict[str, Any]) -> dict[str, Any]:
    prepared = copy.deepcopy(state)
    prepared["analyses"] = []
    prepared["debate_rounds"] = []
    prepared["debate_status"] = {}
    prepared["diagnosis"] = None
    prepared.pop("execution_metrics", None)
    prepared.pop("execution_summary", None)
    prepared.pop("execution_variant", None)
    prepared.pop("model_route", None)
    return prepared


def _execution_summary_from_agent(agent: SingleShotDiagnosisAgent) -> dict[str, Any]:
    metrics = agent.consume_runtime_metrics()
    return {
        "total_node_latency_seconds": None,
        "total_llm_calls": metrics["llm_calls"],
        "calls_by_model": metrics["calls_by_model"],
        "debate_rounds_completed": 0,
        "debate_stopped_by": None,
        "cost_proxy_unit": "llm_call",
        "cost_proxy_value": metrics["llm_calls"],
    }


def _execution_summary_from_agent_metrics(
    *,
    elapsed_seconds: float,
    metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    total_calls = sum(int(metric["llm_calls"]) for metric in metrics)
    calls_by_model: dict[str, int] = {}
    for metric in metrics:
        for model, calls in metric["calls_by_model"].items():
            calls_by_model[model] = calls_by_model.get(model, 0) + int(calls)

    return {
        "total_node_latency_seconds": round(elapsed_seconds, 6),
        "total_llm_calls": total_calls,
        "calls_by_model": calls_by_model,
        "debate_rounds_completed": 0,
        "debate_stopped_by": None,
        "cost_proxy_unit": "llm_call",
        "cost_proxy_value": total_calls,
        "max_concurrency": LLM_MAX_CONCURRENCY,
    }


def _route_for_baseline(
    baseline: BaselineName,
    *,
    single_model: str,
    b3_route_preset: str,
) -> dict[str, str]:
    if baseline in {"b1", "b2"}:
        return {"single": single_model, "preset": baseline}
    if baseline == "b3":
        return get_model_route(b3_route_preset).as_dict()
    if baseline == "b4h":
        return get_model_route("b4h").as_dict()
    return get_model_route("b4r").as_dict()


def _run_single_llm(
    *,
    baseline: BaselineName,
    state: dict[str, Any],
    model: str,
    self_consistency_samples: int,
) -> dict[str, Any]:
    if baseline == "b1":
        agent = SingleShotDiagnosisAgent(model=model, structured_reasoning=False)
        started = time.perf_counter()
        output = agent.run_diagnosis(state)
        elapsed = time.perf_counter() - started
        summary = _execution_summary_from_agent(agent)
        summary["total_node_latency_seconds"] = round(elapsed, 6)
        return {
            "diagnosis": {
                "role": agent.role,
                "model": agent.model,
                "output": output.model_dump(),
            },
            "execution_summary": summary,
            "samples": [],
        }

    started = time.perf_counter()
    if LLM_MAX_CONCURRENCY <= 1 or self_consistency_samples <= 1:
        agent = SingleShotDiagnosisAgent(model=model, structured_reasoning=True, temperature=0.4)
        samples = [agent.run_diagnosis(state) for _ in range(self_consistency_samples)]
        metrics = [agent.consume_runtime_metrics()]
    else:
        max_workers = min(LLM_MAX_CONCURRENCY, self_consistency_samples)

        def run_sample(_index: int):
            sample_agent = SingleShotDiagnosisAgent(
                model=model,
                structured_reasoning=True,
                temperature=0.4,
            )
            sample = sample_agent.run_diagnosis(state)
            return sample, sample_agent.consume_runtime_metrics()

        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for index in range(self_consistency_samples):
                context = copy_context()
                futures.append(executor.submit(context.run, run_sample, index))
            sample_results = [future.result() for future in futures]

        samples = [sample for sample, _ in sample_results]
        metrics = [sample_metrics for _, sample_metrics in sample_results]

    output = aggregate_self_consistency(samples)
    elapsed = time.perf_counter() - started
    summary = _execution_summary_from_agent_metrics(
        elapsed_seconds=elapsed,
        metrics=metrics,
    )
    return {
        "diagnosis": {
            "role": "single_llm_self_consistency",
            "model": model,
            "output": output.model_dump(),
        },
        "execution_summary": summary,
        "samples": [sample.model_dump() for sample in samples],
    }


def _run_graph_baseline(
    *,
    baseline: BaselineName,
    state: dict[str, Any],
    b3_route_preset: str,
) -> dict[str, Any]:
    if baseline == "b3":
        route = get_model_route(b3_route_preset)
        graph = build_graph(routes=route, enable_debate=False, variant="b3")
    elif baseline == "b4h":
        route = get_model_route("b4h")
        graph = build_graph(routes=route, enable_debate=True, variant="b4h")
    else:
        route = get_model_route("b4r")
        graph = build_graph(routes=route, enable_debate=True, variant="b4r")
    return graph.invoke(state, config={"max_concurrency": LLM_MAX_CONCURRENCY})


def run_baseline(
    baseline: BaselineName,
    state: dict[str, Any],
    *,
    case_id: str,
    run_id: str | None = None,
    trace_db_path: str | None = TRACE_DB_PATH,
    single_model: str = DEFAULT_LLM,
    self_consistency_samples: int = 5,
    b3_route_preset: str = "b4r",
    raise_errors: bool = False,
) -> dict[str, Any]:
    if baseline not in BASELINE_NAMES:
        raise ValueError(f"Unknown baseline: {baseline!r}")

    prepared_state = _prepare_state(state)
    resolved_run_id = run_id or f"{case_id}_{baseline}_{uuid.uuid4().hex[:10]}"
    model_route = _route_for_baseline(
        baseline,
        single_model=single_model,
        b3_route_preset=b3_route_preset,
    )
    store = SQLiteTraceStore(trace_db_path) if trace_db_path else None
    if store:
        store.create_run(
            run_id=resolved_run_id,
            baseline=baseline,
            case_id=case_id,
            input_hash=stable_hash(prepared_state),
            model_route=model_route,
            metadata={"self_consistency_samples": self_consistency_samples},
        )

    started = time.perf_counter()
    status = "ok"
    error = None
    result: dict[str, Any] | None = None
    trace_context = (
        trace_run(
            run_id=resolved_run_id,
            baseline=baseline,
            case_id=case_id,
            db_path=trace_db_path,
            metadata={"model_route": model_route},
        )
        if store
        else nullcontext()
    )
    try:
        with trace_context:
            if baseline in {"b1", "b2"}:
                result = _run_single_llm(
                    baseline=baseline,
                    state=prepared_state,
                    model=single_model,
                    self_consistency_samples=self_consistency_samples,
                )
            else:
                result = _run_graph_baseline(
                    baseline=baseline,
                    state=prepared_state,
                    b3_route_preset=b3_route_preset,
                )
    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
        if raise_errors:
            raise

    elapsed = time.perf_counter() - started
    if result is None:
        result = {}

    result.update(
        {
            "run_id": resolved_run_id,
            "baseline": baseline,
            "case_id": case_id,
            "status": status,
            "error": error,
            "model_route": result.get("model_route", model_route),
            "execution_variant": result.get("execution_variant", baseline),
        }
    )

    if store:
        store.record_baseline_result(
            run_id=resolved_run_id,
            baseline=baseline,
            case_id=case_id,
            status=status,
            elapsed_seconds=round(elapsed, 6),
            diagnosis=result.get("diagnosis"),
            execution_summary=result.get("execution_summary"),
            result=result,
            error=error,
        )

    return result
