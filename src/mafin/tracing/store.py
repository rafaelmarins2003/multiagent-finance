from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def stable_hash(value: Any) -> str:
    payload = value if isinstance(value, str) else stable_json(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return stable_json(value)


class SQLiteTraceStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    def ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS experiment_runs (
                    run_id TEXT PRIMARY KEY,
                    baseline TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    input_hash TEXT,
                    model_route_json TEXT,
                    metadata_json TEXT
                );

                CREATE TABLE IF NOT EXISTS llm_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    baseline TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    agent_role TEXT NOT NULL,
                    model TEXT NOT NULL,
                    method TEXT NOT NULL,
                    schema_name TEXT,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    latency_seconds REAL NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    prompt_chars INTEGER NOT NULL,
                    system_prompt_hash TEXT NOT NULL,
                    system_prompt_chars INTEGER NOT NULL,
                    response_chars INTEGER,
                    raw_response_json TEXT,
                    parsed_response_json TEXT,
                    error TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    total_tokens INTEGER,
                    response_metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES experiment_runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS baseline_results (
                    run_id TEXT NOT NULL,
                    baseline TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    elapsed_seconds REAL NOT NULL,
                    diagnosis_json TEXT,
                    execution_summary_json TEXT,
                    result_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, baseline, case_id),
                    FOREIGN KEY(run_id) REFERENCES experiment_runs(run_id)
                );

                CREATE INDEX IF NOT EXISTS idx_llm_calls_run_id ON llm_calls(run_id);
                CREATE INDEX IF NOT EXISTS idx_llm_calls_baseline ON llm_calls(baseline);
                CREATE INDEX IF NOT EXISTS idx_results_baseline ON baseline_results(baseline);
                """
            )

    def create_run(
        self,
        *,
        run_id: str,
        baseline: str,
        case_id: str,
        input_hash: str | None,
        model_route: dict[str, Any] | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.ensure_schema()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiment_runs (
                    run_id, baseline, case_id, created_at, input_hash,
                    model_route_json, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    baseline,
                    case_id,
                    utc_now_iso(),
                    input_hash,
                    _json_or_none(model_route),
                    _json_or_none(metadata),
                ),
            )

    def record_llm_call(
        self,
        *,
        run_id: str,
        baseline: str,
        case_id: str,
        agent_role: str,
        model: str,
        method: str,
        schema_name: str | None,
        status: str,
        started_at: str,
        latency_seconds: float,
        system_prompt: str,
        prompt: str,
        raw_response: Any = None,
        parsed_response: Any = None,
        error: str | None = None,
        usage_metadata: dict[str, Any] | None = None,
        response_metadata: dict[str, Any] | None = None,
    ) -> None:
        usage_metadata = usage_metadata or {}
        raw_response_json = _json_or_none(raw_response)
        response_chars = len(raw_response_json or "")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO llm_calls (
                    run_id, baseline, case_id, agent_role, model, method,
                    schema_name, status, started_at, latency_seconds,
                    prompt_hash, prompt_chars, system_prompt_hash,
                    system_prompt_chars, response_chars, raw_response_json,
                    parsed_response_json, error, input_tokens, output_tokens,
                    total_tokens, response_metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    baseline,
                    case_id,
                    agent_role,
                    model,
                    method,
                    schema_name,
                    status,
                    started_at,
                    latency_seconds,
                    stable_hash(prompt),
                    len(prompt),
                    stable_hash(system_prompt),
                    len(system_prompt),
                    response_chars,
                    raw_response_json,
                    _json_or_none(parsed_response),
                    error,
                    usage_metadata.get("input_tokens"),
                    usage_metadata.get("output_tokens"),
                    usage_metadata.get("total_tokens"),
                    _json_or_none(response_metadata),
                    utc_now_iso(),
                ),
            )

    def record_baseline_result(
        self,
        *,
        run_id: str,
        baseline: str,
        case_id: str,
        status: str,
        elapsed_seconds: float,
        diagnosis: dict[str, Any] | None,
        execution_summary: dict[str, Any] | None,
        result: dict[str, Any] | None,
        error: str | None = None,
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO baseline_results (
                    run_id, baseline, case_id, status, elapsed_seconds,
                    diagnosis_json, execution_summary_json, result_json, error, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    baseline,
                    case_id,
                    status,
                    elapsed_seconds,
                    _json_or_none(diagnosis),
                    _json_or_none(execution_summary),
                    _json_or_none(result),
                    error,
                    utc_now_iso(),
                ),
            )
