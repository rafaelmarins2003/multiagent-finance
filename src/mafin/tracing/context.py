from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mafin.config import TRACE_DB_PATH
from mafin.tracing.store import SQLiteTraceStore


@dataclass(frozen=True)
class TraceContext:
    run_id: str
    baseline: str
    case_id: str
    db_path: Path = Path(TRACE_DB_PATH)
    metadata: dict[str, Any] = field(default_factory=dict)


_TRACE_CONTEXT: ContextVar[TraceContext | None] = ContextVar(
    "mafin_trace_context",
    default=None,
)


def get_trace_context() -> TraceContext | None:
    return _TRACE_CONTEXT.get()


@contextmanager
def trace_run(
    *,
    run_id: str,
    baseline: str,
    case_id: str,
    db_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[TraceContext]:
    context = TraceContext(
        run_id=run_id,
        baseline=baseline,
        case_id=case_id,
        db_path=Path(db_path or TRACE_DB_PATH),
        metadata=metadata or {},
    )
    SQLiteTraceStore(context.db_path).ensure_schema()
    token = _TRACE_CONTEXT.set(context)
    try:
        yield context
    finally:
        _TRACE_CONTEXT.reset(token)
