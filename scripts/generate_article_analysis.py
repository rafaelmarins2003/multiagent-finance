from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

BASELINES = ["b1", "b2", "b3", "b4h", "b4r"]

# Official pay-as-you-go token prices available from model providers on 2026-05-12.
# Values are USD per 1M tokens. DeepSeek uses the active discounted V4-Pro price.
OFFICIAL_TOKEN_PRICES = {
    "deepseek-v4-pro": {"input": 0.435, "output": 0.87},
    "minimax-m2.7": {"input": 0.30, "output": 1.20},
}

# Ollama Cloud exposes usage levels, not per-token prices. This mapping turns
# the official usage levels into a simple relative proxy: medium=2, extra high=4.
OLLAMA_USAGE_LEVEL = {
    "deepseek-v4-pro": 4,
    "minimax-m2.7": 2,
    "nemotron-3-super": 2,
}


def normalize_case_id(case_id: str) -> str:
    return case_id.removesuffix("_retry")


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def load_result_rows(result_dir: Path) -> list[dict]:
    result_files = [
        *sorted(result_dir.glob("enriched_20260512T054520Z_*.json")),
        result_dir / "retry_enriched_xp_top_acoes_b4h.json",
        *sorted(result_dir.glob("enriched_real_full_remaining_retry_20260512T131305Z_*.json")),
    ]
    rows: list[dict] = []
    for path in result_files:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data.get("results", []):
            if row.get("status") != "ok":
                continue
            normalized = dict(row)
            normalized["raw_case_id"] = normalized["case_id"]
            normalized["case_id"] = normalize_case_id(normalized["case_id"])
            rows.append(normalized)
    return rows


def keep_completed_cases(rows: list[dict]) -> dict[str, dict[str, dict]]:
    by_case: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_case[row["case_id"]][row["baseline"]] = row
    return {
        case_id: baselines
        for case_id, baselines in by_case.items()
        if all(baseline in baselines for baseline in BASELINES)
    }


def load_trace_calls(trace_dir: Path, run_ids: set[str]) -> list[dict]:
    trace_files = [
        *sorted(trace_dir.glob("enriched_20260512T054520Z_*.sqlite3")),
        trace_dir / "retry_enriched_xp_top_acoes_b4h.sqlite3",
        *sorted(trace_dir.glob("enriched_real_full_remaining_retry_20260512T131305Z_*.sqlite3")),
    ]
    calls: list[dict] = []
    for path in trace_files:
        if not path.exists():
            continue
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        for row in connection.execute("select * from llm_calls"):
            if row["run_id"] in run_ids:
                calls.append(dict(row))
        connection.close()
    return calls


def diagnosis_output(row: dict) -> dict:
    diagnosis = row.get("diagnosis") or {}
    return diagnosis.get("output") or {}


def consensus_by_case(completed_cases: dict[str, dict[str, dict]]) -> dict[str, str | None]:
    consensus: dict[str, str | None] = {}
    for case_id, rows in completed_cases.items():
        labels = [diagnosis_output(row).get("classification") for row in rows.values()]
        counts = Counter(labels)
        top = counts.most_common()
        consensus[case_id] = top[0][0] if len(top) == 1 or top[0][1] > top[1][1] else None
    return consensus


def compute_metrics(
    completed_cases: dict[str, dict[str, dict]],
    calls: list[dict],
) -> dict[str, dict]:
    rows = [row for baselines in completed_cases.values() for row in baselines.values()]
    consensus = consensus_by_case(completed_cases)
    metrics: dict[str, dict] = {}

    for baseline in BASELINES:
        result_rows = [row for row in rows if row["baseline"] == baseline]
        baseline_calls = [call for call in calls if call["baseline"] == baseline]
        ok_calls = [call for call in baseline_calls if call["status"] == "ok"]

        labels: list[str] = []
        detail_counts: list[int] = []
        justification_words: list[int] = []
        consensus_total = 0
        consensus_hits = 0

        for row in result_rows:
            output = diagnosis_output(row)
            label = output.get("classification")
            labels.append(label)
            positive_factors = output.get("positive_factors") or []
            negative_factors = output.get("negative_factors") or []
            detail_counts.append(
                len(positive_factors) + len(negative_factors)
            )
            justification_words.append(len(str(output.get("justification") or "").split()))
            case_consensus = consensus.get(row["case_id"])
            if case_consensus is not None:
                consensus_total += 1
                consensus_hits += int(label == case_consensus)

        calls_by_model: Counter[str] = Counter()
        tokens_by_model: Counter[str] = Counter()
        official_cost_by_model: Counter[str] = Counter()
        official_cost_usd = 0.0
        unpriced_tokens = 0
        weighted_usage_units = 0

        for call in ok_calls:
            model = call["model"]
            input_tokens = call["input_tokens"] or 0
            output_tokens = call["output_tokens"] or 0
            total_tokens = input_tokens + output_tokens
            calls_by_model[model] += 1
            tokens_by_model[model] += total_tokens
            weighted_usage_units += OLLAMA_USAGE_LEVEL.get(model, 1) * total_tokens
            price = OFFICIAL_TOKEN_PRICES.get(model)
            if price is None:
                unpriced_tokens += total_tokens
                continue
            cost = (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000
            official_cost_usd += cost
            official_cost_by_model[model] += cost

        latencies = [
            float(row["execution_summary"].get("total_node_latency_seconds") or 0)
            for row in result_rows
        ]
        total_tokens = sum(call["total_tokens"] or 0 for call in ok_calls)

        metrics[baseline] = {
            "cases": len(result_rows),
            "classifications": dict(Counter(labels)),
            "conclusive_rate": sum(label != "inconclusiva" for label in labels) / len(labels),
            "alert_rate": sum(
                label in {"atencao", "risco_elevado", "desalinhada"} for label in labels
            )
            / len(labels),
            "consensus_accuracy_proxy": consensus_hits / consensus_total,
            "consensus_cases": consensus_total,
            "detail_factors_avg": mean(detail_counts),
            "justification_words_avg": mean(justification_words),
            "total_calls": len(baseline_calls),
            "ok_calls": len(ok_calls),
            "call_errors": sum(call["status"] != "ok" for call in baseline_calls),
            "latency_avg_seconds": mean(latencies),
            "latency_p50_seconds": percentile(latencies, 0.50),
            "latency_p95_seconds": percentile(latencies, 0.95),
            "tokens_per_case": total_tokens / len(result_rows),
            "official_priced_cost_usd": official_cost_usd,
            "official_priced_cost_per_case_usd": official_cost_usd / len(result_rows),
            "unpriced_tokens": unpriced_tokens,
            "ollama_usage_units_per_case": weighted_usage_units / len(result_rows),
            "calls_by_model": dict(calls_by_model),
            "tokens_by_model": dict(tokens_by_model),
            "official_cost_by_model": dict(official_cost_by_model),
        }
    return metrics


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_simple_bar_pdf(path: Path, title: str, xlabel: str, values: dict[str, float]) -> None:
    width = 540
    height = 310
    left = 95
    right = 35
    top = 64
    bar_h = 24
    gap = 20
    max_value = max(values.values()) or 1

    commands = [
        "1 1 1 rg 0 0 540 310 re f",
        "0 0 0 rg /F1 15 Tf",
        f"BT 35 282 Td ({pdf_escape(title)}) Tj ET",
        "/F1 9 Tf",
        f"BT 35 21 Td ({pdf_escape(xlabel)}) Tj ET",
    ]

    for index, (label, value) in enumerate(values.items()):
        y = height - top - index * (bar_h + gap)
        bar_w = (width - left - right) * (value / max_value)
        commands.extend(
            [
                "0 0 0 rg /F1 10 Tf",
                f"BT 35 {y + 7:.1f} Td ({pdf_escape(label.upper())}) Tj ET",
                "0.90 0.92 0.95 rg",
                f"{left} {y:.1f} {width - left - right} {bar_h} re f",
                "0.16 0.33 0.54 rg",
                f"{left} {y:.1f} {bar_w:.1f} {bar_h} re f",
                "0 0 0 rg /F1 9 Tf",
                f"BT {left + bar_w + 6:.1f} {y + 7:.1f} Td ({value:.1f}) Tj ET",
            ]
        )

    stream = "\n".join(commands).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 540 310] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        (
            b"<< /Length "
            + str(len(stream)).encode("ascii")
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        ),
    ]

    chunks = [b"%PDF-1.4\n"]
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(sum(len(chunk) for chunk in chunks))
        chunks.append(f"{number} 0 obj\n".encode("ascii") + obj + b"\nendobj\n")
    xref = sum(len(chunk) for chunk in chunks)
    chunks.append(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    chunks.append(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        chunks.append(f"{offset:010d} 00000 n \n".encode("ascii"))
    chunks.append(
        (
            f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(b"".join(chunks))


def write_outputs(
    metrics: dict[str, dict],
    output_json: Path,
    output_csv: Path,
    figures_dir: Path,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = [
        "baseline",
        "cases",
        "conclusive_rate",
        "consensus_accuracy_proxy",
        "detail_factors_avg",
        "justification_words_avg",
        "latency_p50_seconds",
        "latency_p95_seconds",
        "tokens_per_case",
        "ollama_usage_units_per_case",
        "official_priced_cost_usd",
        "official_priced_cost_per_case_usd",
        "unpriced_tokens",
        "total_calls",
        "call_errors",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for baseline in BASELINES:
            row = {"baseline": baseline}
            row.update({key: metrics[baseline][key] for key in fieldnames if key != "baseline"})
            writer.writerow(row)

    write_simple_bar_pdf(
        figures_dir / "cost_proxy.pdf",
        "Custo operacional proxy por caso",
        "Unidades relativas = tokens x nivel oficial de uso no Ollama Cloud",
        {
            baseline: metrics[baseline]["ollama_usage_units_per_case"] / 1000
            for baseline in BASELINES
        },
    )
    write_simple_bar_pdf(
        figures_dir / "accuracy_proxy.pdf",
        "Acuracia proxy por consenso",
        "Percentual de concordancia com a maioria dos baselines; empates excluidos",
        {baseline: metrics[baseline]["consensus_accuracy_proxy"] * 100 for baseline in BASELINES},
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result_rows = load_result_rows(root / "data" / "results")
    completed_cases = keep_completed_cases(result_rows)
    run_ids = {
        row["run_id"] for baselines in completed_cases.values() for row in baselines.values()
    }
    calls = load_trace_calls(root / "data" / "traces", run_ids)
    metrics = compute_metrics(completed_cases, calls)
    write_outputs(
        metrics,
        root / "data" / "results" / "article_comparison_metrics.json",
        root / "data" / "results" / "article_comparison_metrics.csv",
        root / "article" / "my_template" / "figures",
    )
    print(f"Completed enriched cases: {len(completed_cases)}")
    print(f"Baselines: {', '.join(BASELINES)}")
    print("Wrote data/results/article_comparison_metrics.json")
    print("Wrote data/results/article_comparison_metrics.csv")
    print("Wrote article/my_template/figures/cost_proxy.pdf")
    print("Wrote article/my_template/figures/accuracy_proxy.pdf")


if __name__ == "__main__":
    main()
