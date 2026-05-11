from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from mafin.config import get_model_route
from mafin.llm.ollama import make_chat


class CompatibilityOutput(BaseModel):
    status: Literal["ok", "limited"]
    summary: str
    factors: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark structured-output compatibility for candidate Ollama models."
    )
    parser.add_argument(
        "--route-preset",
        choices=["env", "local", "b4h", "b4r"],
        default="b4r",
        help="Route used to select unique models when --models is omitted.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional explicit model list. Defaults to unique models in the selected route.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def benchmark_model(model: str, temperature: float) -> dict:
    llm = make_chat(model=model, temperature=temperature)
    structured = llm.with_structured_output(CompatibilityOutput, method="json_schema")
    started = time.perf_counter()
    try:
        output = structured.invoke(
            [
                SystemMessage(
                    content=(
                        "Você testa compatibilidade de saída estruturada. "
                        "Responda apenas pelo schema solicitado."
                    )
                ),
                HumanMessage(
                    content=(
                        "Dado PETR4 com preço abaixo da média de 50 dias e carteira "
                        "moderada, produza um diagnóstico curto de compatibilidade."
                    )
                ),
            ]
        )
        elapsed = time.perf_counter() - started
        return {
            "model": model,
            "ok": True,
            "elapsed_seconds": round(elapsed, 6),
            "output": output.model_dump(),
            "error": None,
        }
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return {
            "model": model,
            "ok": False,
            "elapsed_seconds": round(elapsed, 6),
            "output": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    args = parse_args()
    route = get_model_route(args.route_preset)
    models = args.models if args.models else route.unique_models()
    results = [benchmark_model(model, args.temperature) for model in models]

    payload = {
        "route_preset": route.preset,
        "models": models,
        "results": results,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
