import json
import random
import re
import threading
import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from json import JSONDecodeError
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from mafin.config import (
    DEFAULT_TEMPERATURE,
    STRUCTURED_OUTPUT_MAX_RETRIES,
    STRUCTURED_OUTPUT_MODE,
    STRUCTURED_OUTPUT_RETRY_BASE_SECONDS,
    STRUCTURED_OUTPUT_RETRY_MAX_SECONDS,
)
from mafin.llm.ollama import make_chat
from mafin.tracing import SQLiteTraceStore, get_trace_context


def _extract_json_object(text: str) -> str:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
            return json.dumps(parsed, ensure_ascii=False)
        except JSONDecodeError:
            continue
    raise ValueError("No JSON object found in model response.")


def _message_to_jsonable(message: Any) -> dict[str, Any] | str | None:
    if message is None:
        return None
    if hasattr(message, "model_dump"):
        return message.model_dump(mode="json")
    return str(message)


def _model_to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _parse_structured_json(content: str, schema: type[BaseModel]) -> BaseModel:
    json_content = _extract_json_object(content)
    try:
        return schema.model_validate_json(json_content)
    except ValidationError:
        raw = json.loads(json_content)
        coerced = _coerce_single_per_ticker_output(raw, schema)
        if coerced is None:
            raise
        return schema.model_validate(coerced)


def _coerce_single_per_ticker_output(
    raw: Any,
    schema: type[BaseModel],
) -> dict[str, Any] | None:
    fields = getattr(schema, "model_fields", {})
    if "per_ticker" not in fields or "overall" not in fields:
        return None
    if isinstance(raw, list):
        return {
            "per_ticker": raw,
            "overall": "Resumo não fornecido explicitamente pelo modelo.",
        }
    if not isinstance(raw, dict):
        return None
    if "per_ticker" in raw or "overall" in raw:
        return None
    if "ticker" not in raw:
        return None
    return {
        "per_ticker": [raw],
        "overall": raw.get("summary") or "Resumo não fornecido explicitamente pelo modelo.",
    }


def _usage_metadata(message: Any) -> dict[str, Any]:
    usage = getattr(message, "usage_metadata", None)
    return dict(usage) if usage else {}


def _response_metadata(message: Any) -> dict[str, Any]:
    metadata = getattr(message, "response_metadata", None)
    return dict(metadata) if metadata else {}


def _retry_after_seconds(error: Exception) -> float | None:
    text = f"{type(error).__name__}: {error}".lower()
    if not any(
        marker in text
        for marker in (
            "status code: 429",
            "status code: 500",
            "status code: 502",
            "status code: 503",
            "status code: 504",
            "temporarily overloaded",
            "timeout",
            "timed out",
        )
    ):
        return None

    match = re.search(r"retry(?:-|\s*)after[:=]\s*(\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1))
    return STRUCTURED_OUTPUT_RETRY_BASE_SECONDS


def _sleep_before_retry(error: Exception, attempt: int) -> None:
    base_delay = _retry_after_seconds(error)
    if base_delay is None or base_delay <= 0:
        return
    delay = min(STRUCTURED_OUTPUT_RETRY_MAX_SECONDS, base_delay * (2 ** (attempt - 1)))
    delay += random.uniform(0.0, min(3.0, delay * 0.1))
    print(
        f"[{threading.current_thread().name}] transient LLM error; "
        f"retrying in {delay:.1f}s ({type(error).__name__})",
        flush=True,
    )
    time.sleep(delay)


class Agent(ABC):
    role: str
    system_prompt: str

    def __init__(self, model: str, temperature: float = DEFAULT_TEMPERATURE):
        self.model = model
        self.llm = make_chat(model=model, temperature=temperature)
        self._llm_call_count = 0

    def reset_runtime_metrics(self) -> None:
        self._llm_call_count = 0

    def consume_runtime_metrics(self) -> dict[str, Any]:
        call_count = self._llm_call_count
        self._llm_call_count = 0
        return {
            "llm_calls": call_count,
            "calls_by_model": {self.model: call_count} if call_count else {},
        }

    def _invoke(self, user_prompt: str) -> str:
        self._llm_call_count += 1
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt),
        ]
        started_at = datetime.now(UTC).isoformat()
        started = time.perf_counter()
        try:
            msg = self.llm.invoke(messages)
            elapsed = time.perf_counter() - started
            self._trace_llm_call(
                method="raw",
                schema_name=None,
                status="ok",
                started_at=started_at,
                latency_seconds=elapsed,
                user_prompt=user_prompt,
                raw_response=_message_to_jsonable(msg),
                usage_metadata=_usage_metadata(msg),
                response_metadata=_response_metadata(msg),
            )
            return msg.content if isinstance(msg.content, str) else str(msg.content)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            self._trace_llm_call(
                method="raw",
                schema_name=None,
                status="error",
                started_at=started_at,
                latency_seconds=elapsed,
                user_prompt=user_prompt,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

    def _invoke_structured(self, user_prompt: str, schema: type[BaseModel]):
        if STRUCTURED_OUTPUT_MODE == "manual":
            return self._invoke_structured_manual(user_prompt, schema)

        self._llm_call_count += 1
        structured_llm = self.llm.with_structured_output(
            schema,
            method="json_schema",
            include_raw=True,
        )
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt),
        ]
        started_at = datetime.now(UTC).isoformat()
        started = time.perf_counter()
        structured_recorded = False
        try:
            result = structured_llm.invoke(messages)
            elapsed = time.perf_counter() - started
            raw = result.get("raw") if isinstance(result, dict) else None
            parsed = result.get("parsed") if isinstance(result, dict) else result
            parsing_error = result.get("parsing_error") if isinstance(result, dict) else None
            if parsing_error is not None or parsed is None:
                structured_recorded = True
                self._trace_llm_call(
                    method="structured_json_schema",
                    schema_name=schema.__name__,
                    status="parse_error",
                    started_at=started_at,
                    latency_seconds=elapsed,
                    user_prompt=user_prompt,
                    raw_response=_message_to_jsonable(raw),
                    error=(
                        f"{type(parsing_error).__name__}: {parsing_error}"
                        if parsing_error
                        else "Parsed output is empty."
                    ),
                    usage_metadata=_usage_metadata(raw),
                    response_metadata=_response_metadata(raw),
                )
                raise parsing_error or ValueError("Parsed output is empty.")

            structured_recorded = True
            self._trace_llm_call(
                method="structured_json_schema",
                schema_name=schema.__name__,
                status="ok",
                started_at=started_at,
                latency_seconds=elapsed,
                user_prompt=user_prompt,
                raw_response=_message_to_jsonable(raw),
                parsed_response=_model_to_jsonable(parsed),
                usage_metadata=_usage_metadata(raw),
                response_metadata=_response_metadata(raw),
            )
            return parsed
        except Exception as first_error:
            if not structured_recorded:
                elapsed = time.perf_counter() - started
                self._trace_llm_call(
                    method="structured_json_schema",
                    schema_name=schema.__name__,
                    status="error",
                    started_at=started_at,
                    latency_seconds=elapsed,
                    user_prompt=user_prompt,
                    error=f"{type(first_error).__name__}: {first_error}",
                )

            schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
            strict_prompt = (
                f"{user_prompt}\n\n"
                "A resposta anterior não respeitou o schema esperado. "
                "Responda agora somente com um objeto JSON válido, sem Markdown, "
                "sem comentários e sem campos fora do schema abaixo.\n\n"
                f"JSON Schema:\n{schema_json}"
            )
            self._llm_call_count += 1
            fallback_started_at = datetime.now(UTC).isoformat()
            fallback_started = time.perf_counter()
            msg = None
            content = ""
            try:
                msg = self.llm.invoke(
                    [
                        SystemMessage(content=self.system_prompt),
                        HumanMessage(content=strict_prompt),
                    ]
                )
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                parsed = _parse_structured_json(content, schema)
                elapsed = time.perf_counter() - fallback_started
                self._trace_llm_call(
                    method="structured_json_fallback",
                    schema_name=schema.__name__,
                    status="ok",
                    started_at=fallback_started_at,
                    latency_seconds=elapsed,
                    user_prompt=strict_prompt,
                    raw_response=_message_to_jsonable(msg),
                    parsed_response=_model_to_jsonable(parsed),
                    usage_metadata=_usage_metadata(msg),
                    response_metadata=_response_metadata(msg),
                )
                return parsed
            except Exception as fallback_error:
                elapsed = time.perf_counter() - fallback_started
                self._trace_llm_call(
                    method="structured_json_fallback",
                    schema_name=schema.__name__,
                    status="error",
                    started_at=fallback_started_at,
                    latency_seconds=elapsed,
                    user_prompt=strict_prompt,
                    raw_response=_message_to_jsonable(msg) or content,
                    error=f"{type(fallback_error).__name__}: {fallback_error}",
                    usage_metadata=_usage_metadata(msg),
                    response_metadata=_response_metadata(msg),
                )
                raise RuntimeError(
                    "Failed to parse structured output for "
                    f"{self.role}. First parser error: {type(first_error).__name__}."
                ) from fallback_error

    def _invoke_structured_manual(self, user_prompt: str, schema: type[BaseModel]):
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        base_prompt = (
            f"{user_prompt}\n\n"
            "Responda somente com um objeto JSON válido, sem Markdown, sem comentários "
            "e sem campos fora do schema abaixo. Use respostas concisas; quando houver listas, "
            "limite a no máximo 5 itens por campo.\n\n"
            f"JSON Schema:\n{schema_json}"
        )

        last_error: Exception | None = None
        for attempt in range(1, STRUCTURED_OUTPUT_MAX_RETRIES + 1):
            strict_prompt = base_prompt
            if attempt > 1:
                strict_prompt = (
                    f"{base_prompt}\n\n"
                    "A tentativa anterior falhou no parsing. Corrija estritamente: "
                    "retorne apenas um objeto JSON completo, fechado e válido."
                )

            self._llm_call_count += 1
            started_at = datetime.now(UTC).isoformat()
            started = time.perf_counter()
            msg = None
            content = ""
            try:
                msg = self.llm.invoke(
                    [
                        SystemMessage(content=self.system_prompt),
                        HumanMessage(content=strict_prompt),
                    ]
                )
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                parsed = _parse_structured_json(content, schema)
                elapsed = time.perf_counter() - started
                self._trace_llm_call(
                    method="structured_json_manual",
                    schema_name=schema.__name__,
                    status="ok",
                    started_at=started_at,
                    latency_seconds=elapsed,
                    user_prompt=strict_prompt,
                    raw_response=_message_to_jsonable(msg),
                    parsed_response=_model_to_jsonable(parsed),
                    usage_metadata=_usage_metadata(msg),
                    response_metadata=_response_metadata(msg),
                )
                return parsed
            except Exception as exc:
                last_error = exc
                elapsed = time.perf_counter() - started
                self._trace_llm_call(
                    method="structured_json_manual",
                    schema_name=schema.__name__,
                    status="retry_error" if attempt < STRUCTURED_OUTPUT_MAX_RETRIES else "error",
                    started_at=started_at,
                    latency_seconds=elapsed,
                    user_prompt=strict_prompt,
                    raw_response=_message_to_jsonable(msg) or content,
                    error=f"{type(exc).__name__}: {exc}",
                    usage_metadata=_usage_metadata(msg),
                    response_metadata=_response_metadata(msg),
                )
                if attempt < STRUCTURED_OUTPUT_MAX_RETRIES:
                    _sleep_before_retry(exc, attempt)

        raise RuntimeError(f"Failed to parse structured output for {self.role}.") from last_error

    def _trace_llm_call(
        self,
        *,
        method: str,
        schema_name: str | None,
        status: str,
        started_at: str,
        latency_seconds: float,
        user_prompt: str,
        raw_response: Any = None,
        parsed_response: Any = None,
        error: str | None = None,
        usage_metadata: dict[str, Any] | None = None,
        response_metadata: dict[str, Any] | None = None,
    ) -> None:
        context = get_trace_context()
        if context is None:
            return
        SQLiteTraceStore(context.db_path).record_llm_call(
            run_id=context.run_id,
            baseline=context.baseline,
            case_id=context.case_id,
            agent_role=self.role,
            model=self.model,
            method=method,
            schema_name=schema_name,
            status=status,
            started_at=started_at,
            latency_seconds=round(latency_seconds, 6),
            system_prompt=self.system_prompt,
            prompt=user_prompt,
            raw_response=raw_response,
            parsed_response=parsed_response,
            error=error,
            usage_metadata=usage_metadata,
            response_metadata=response_metadata,
        )

    @abstractmethod
    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Read state, call LLM, return partial state update."""
