import sqlite3

from mafin.agents.base import _parse_structured_json
from mafin.agents.fundamental import _normalize_fundamental_output
from mafin.baselines.single import aggregate_self_consistency
from mafin.schema import FundamentalAnalysisOutput, PortfolioDiagnosisOutput
from mafin.tracing.store import SQLiteTraceStore
from scripts.run_baselines import _cases_from_workload, summarize_results


def _diagnosis(classification: str, confidence: float) -> PortfolioDiagnosisOutput:
    return PortfolioDiagnosisOutput(
        classification=classification,
        justification=f"justificativa {classification}",
        positive_factors=["diversificação"],
        negative_factors=["volatilidade"],
        confidence=confidence,
        profile_alignment="alinhamento parcial",
    )


def test_self_consistency_aggregates_majority_classification():
    output = aggregate_self_consistency(
        [
            _diagnosis("atencao", 0.7),
            _diagnosis("atencao", 0.8),
            _diagnosis("risco_elevado", 0.95),
        ]
    )

    assert output.classification == "atencao"
    assert output.confidence == 0.75
    assert "2/3" in output.justification


def test_sqlite_trace_store_records_run_call_and_result(tmp_path):
    db_path = tmp_path / "trace.sqlite3"
    store = SQLiteTraceStore(db_path)
    store.create_run(
        run_id="run-1",
        baseline="b1",
        case_id="case-1",
        input_hash="abc",
        model_route={"single": "model"},
    )
    store.record_llm_call(
        run_id="run-1",
        baseline="b1",
        case_id="case-1",
        agent_role="single_llm",
        model="model",
        method="structured_json_schema",
        schema_name="PortfolioDiagnosisOutput",
        status="ok",
        started_at="2026-05-10T00:00:00+00:00",
        latency_seconds=1.23,
        system_prompt="system",
        prompt="prompt",
        raw_response={"content": "{}"},
        parsed_response={"classification": "atencao"},
    )
    store.record_baseline_result(
        run_id="run-1",
        baseline="b1",
        case_id="case-1",
        status="ok",
        elapsed_seconds=1.25,
        diagnosis={"output": {"classification": "atencao"}},
        execution_summary={"total_llm_calls": 1},
        result={"status": "ok"},
    )

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM experiment_runs").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM baseline_results").fetchone()[0] == 1


def test_parse_structured_json_wraps_single_per_ticker_item():
    output = _parse_structured_json(
        '{"ticker":"PETR4","summary":"dados disponíveis","strengths":["lucro"]}',
        FundamentalAnalysisOutput,
    )

    assert output.per_ticker[0].ticker == "PETR4"
    assert output.overall == "dados disponíveis"


def test_normalize_fundamental_output_fills_missing_tickers():
    output = FundamentalAnalysisOutput.model_validate(
        {
            "per_ticker": [
                {"ticker": "PETR4", "summary": "dados disponíveis", "strengths": ["lucro"]}
            ],
            "overall": "resumo",
        }
    )

    normalized = _normalize_fundamental_output(
        output,
        [
            {"ticker": "PETR4", "weight": 0.6, "sector": "Energia"},
            {"ticker": "VALE3", "weight": 0.4, "sector": "Mineração"},
        ],
    )

    assert [item.ticker for item in normalized.per_ticker] == ["PETR4", "VALE3"]
    assert normalized.per_ticker[1].data_gaps


def test_workload_cases_are_converted_to_graph_state(tmp_path):
    workload_path = tmp_path / "workload.json"
    workload_path.write_text(
        """
        {
          "cases": [
            {
              "case_id": "case-1",
              "portfolio": [{"ticker": "PETR4", "weight": 1.0, "sector": "Energia"}],
              "profile": {"risk_tolerance": "moderate"}
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    cases = _cases_from_workload(workload_path, case_id=None, offset_cases=0, limit_cases=None)

    assert cases[0]["case_id"] == "case-1"
    assert cases[0]["input_kind"] == "workload"
    assert cases[0]["state"]["portfolio"][0]["ticker"] == "PETR4"
    assert cases[0]["state"]["market_data"] == {}


def test_portfolio_collection_is_converted_to_workload_cases(tmp_path):
    workload_path = tmp_path / "portfolios.json"
    workload_path.write_text(
        """
        {
          "portfolios": [
            {
              "id": "broker-case-1",
              "broker": "Broker",
              "portfolio": [{"ticker": "PETR4", "weight": 1.0, "sector": "Energia"}],
              "profile": {"risk_tolerance": "moderate", "horizon": "long", "objective": "teste"}
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    cases = _cases_from_workload(workload_path, case_id=None, offset_cases=0, limit_cases=None)

    assert cases[0]["case_id"] == "broker-case-1"
    assert cases[0]["metadata"]["broker"] == "Broker"
    assert cases[0]["state"]["profile"]["risk_tolerance"] == "moderate"


def test_workload_cases_support_offset_and_limit(tmp_path):
    workload_path = tmp_path / "workload.json"
    workload_path.write_text(
        """
        {
          "cases": [
            {
              "case_id": "case-1",
              "portfolio": [{"ticker": "PETR4", "weight": 1.0, "sector": "Energia"}],
              "profile": {"risk_tolerance": "moderate"}
            },
            {
              "case_id": "case-2",
              "portfolio": [{"ticker": "VALE3", "weight": 1.0, "sector": "Mineração"}],
              "profile": {"risk_tolerance": "moderate"}
            },
            {
              "case_id": "case-3",
              "portfolio": [{"ticker": "ITUB4", "weight": 1.0, "sector": "Financeiro"}],
              "profile": {"risk_tolerance": "moderate"}
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    cases = _cases_from_workload(
        workload_path,
        case_id=None,
        offset_cases=1,
        limit_cases=1,
    )

    assert [case["case_id"] for case in cases] == ["case-2"]


def test_summarize_results_groups_status_classification_and_calls():
    summary = summarize_results(
        [
            {
                "baseline": "b1",
                "status": "ok",
                "diagnosis": {"output": {"classification": "atencao"}},
                "execution_summary": {"total_llm_calls": 2},
            },
            {
                "baseline": "b1",
                "status": "error",
                "diagnosis": None,
                "execution_summary": {"total_llm_calls": 1},
            },
        ]
    )

    assert summary["total_results"] == 2
    assert summary["status_by_baseline"]["b1"] == {"ok": 1, "error": 1}
    assert summary["classification_by_baseline"]["b1"] == {"atencao": 1}
    assert summary["total_llm_calls_by_baseline"]["b1"] == 3
