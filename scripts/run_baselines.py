"""Run experimental baselines over frozen snapshots or workload cases."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from mafin.baselines import BASELINE_NAMES, run_baseline
from mafin.baselines.runner import BaselineName
from mafin.config import DEFAULT_LLM, TRACE_DB_PATH
from mafin.data.snapshot import PortfolioDataSnapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--snapshot", type=Path)
    input_group.add_argument("--workload", type=Path)
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=BASELINE_NAMES,
        default=list(BASELINE_NAMES),
    )
    parser.add_argument("--case-id", default=None, help="Run only this case id.")
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--single-model", default=DEFAULT_LLM)
    parser.add_argument("--self-consistency-samples", type=int, default=5)
    parser.add_argument(
        "--b3-route-preset",
        choices=["env", "local", "b4h", "b4r"],
        default="b4r",
        help="Model route used by B3, which has specialists but no debate.",
    )
    parser.add_argument("--trace-db", type=Path, default=Path(TRACE_DB_PATH))
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def default_output_path() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("data/results") / f"baselines_{timestamp}.json"


def _empty_graph_state(portfolio: list[dict[str, Any]], profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "portfolio": portfolio,
        "profile": profile,
        "market_data": {},
        "fundamental_data": {},
        "sentiment_data": {},
        "macro_data": {},
        "analyses": [],
        "debate_rounds": [],
        "debate_status": {},
        "diagnosis": None,
    }


def _case_from_snapshot(snapshot_path: Path, case_id: str | None) -> list[dict[str, Any]]:
    snapshot = PortfolioDataSnapshot.model_validate_json(
        snapshot_path.read_text(encoding="utf-8")
    )
    return [
        {
            "case_id": case_id or snapshot.snapshot_id,
            "source": str(snapshot_path),
            "input_kind": "snapshot",
            "state": snapshot.to_graph_state(),
        }
    ]


def _cases_from_workload(
    workload_path: Path,
    *,
    case_id: str | None,
    limit_cases: int | None,
) -> list[dict[str, Any]]:
    workload = json.loads(workload_path.read_text(encoding="utf-8"))
    raw_cases = workload.get("cases") or [
        {
            "case_id": portfolio["id"],
            "portfolio": portfolio["portfolio"],
            "profile": portfolio["profile"],
            "metadata": {
                key: value
                for key, value in portfolio.items()
                if key not in {"portfolio", "profile"}
            },
        }
        for portfolio in workload.get("portfolios", [])
    ]
    cases = []
    for case in raw_cases:
        if case_id and case.get("case_id") != case_id:
            continue
        cases.append(
            {
                "case_id": case["case_id"],
                "source": str(workload_path),
                "input_kind": "workload",
                "state": _empty_graph_state(case["portfolio"], case["profile"]),
                "metadata": case.get("metadata", {}),
            }
        )
        if limit_cases is not None and len(cases) >= limit_cases:
            break
    return cases


def load_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.snapshot:
        cases = _case_from_snapshot(args.snapshot, args.case_id)
    else:
        cases = _cases_from_workload(
            args.workload,
            case_id=args.case_id,
            limit_cases=args.limit_cases,
        )

    if not cases:
        raise ValueError("No cases matched the provided input/filter.")
    return cases


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    status_by_baseline: dict[str, Counter] = defaultdict(Counter)
    classification_by_baseline: dict[str, Counter] = defaultdict(Counter)
    total_calls_by_baseline: Counter = Counter()

    for result in results:
        baseline = result["baseline"]
        status_by_baseline[baseline][result["status"]] += 1
        diagnosis = result.get("diagnosis") or {}
        output = diagnosis.get("output") or {}
        classification = output.get("classification")
        if classification:
            classification_by_baseline[baseline][classification] += 1
        summary = result.get("execution_summary") or {}
        total_calls_by_baseline[baseline] += int(summary.get("total_llm_calls") or 0)

    return {
        "total_results": len(results),
        "status_by_baseline": {
            baseline: dict(counts) for baseline, counts in status_by_baseline.items()
        },
        "classification_by_baseline": {
            baseline: dict(counts)
            for baseline, counts in classification_by_baseline.items()
        },
        "total_llm_calls_by_baseline": dict(total_calls_by_baseline),
    }


def main() -> None:
    args = parse_args()
    baselines = [cast(BaselineName, baseline) for baseline in args.baselines]
    output_path = args.output or default_output_path()
    cases = load_cases(args)
    trace_db_path = None if args.no_trace else str(args.trace_db)

    plan = {
        "input": str(args.snapshot or args.workload),
        "input_kind": "snapshot" if args.snapshot else "workload",
        "case_id_filter": args.case_id,
        "case_count": len(cases),
        "case_ids": [case["case_id"] for case in cases],
        "baselines": baselines,
        "single_model": args.single_model,
        "self_consistency_samples": args.self_consistency_samples,
        "b3_route_preset": args.b3_route_preset,
        "trace_db": trace_db_path,
        "output": str(output_path),
    }
    if args.dry_run:
        print(json.dumps({"dry_run": True, "plan": plan}, ensure_ascii=False, indent=2))
        return

    results = []
    for case in cases:
        for baseline in baselines:
            print(f"Running {baseline} on {case['case_id']}...")
            result = run_baseline(
                baseline,
                case["state"],
                case_id=case["case_id"],
                trace_db_path=trace_db_path,
                single_model=args.single_model,
                self_consistency_samples=args.self_consistency_samples,
                b3_route_preset=args.b3_route_preset,
            )
            result["input_kind"] = case["input_kind"]
            result["source"] = case["source"]
            result["case_metadata"] = case.get("metadata", {})
            results.append(result)
            print(f"{case['case_id']} / {baseline}: {result['status']}")

    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "plan": plan,
        "summary": summarize_results(results),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote results to {output_path}")
    if trace_db_path:
        print(f"Wrote trace DB to {trace_db_path}")


if __name__ == "__main__":
    main()
