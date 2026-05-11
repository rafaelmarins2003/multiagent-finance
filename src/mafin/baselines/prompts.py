from __future__ import annotations

import json
from typing import Any

from mafin.data.compact import (
    compact_fundamental_data,
    compact_macro_data,
    compact_market_data,
    compact_sentiment_data,
)


def compact_diagnosis_input(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "portfolio": state.get("portfolio", []),
        "profile": state.get("profile", {}),
        "market_data": compact_market_data(state.get("market_data", {})),
        "fundamental_data": compact_fundamental_data(state.get("fundamental_data", {})),
        "sentiment_data": compact_sentiment_data(state.get("sentiment_data", {})),
        "macro_data": compact_macro_data(state.get("macro_data", {})),
    }


def render_diagnosis_prompt(state: dict[str, Any], *, structured_reasoning: bool) -> str:
    payload = compact_diagnosis_input(state)
    reasoning_instruction = (
        "Antes de decidir, avalie de forma estruturada: alinhamento ao perfil, "
        "concentração, dados técnicos, fundamentos, sentimento, macro e lacunas. "
        "Não exponha raciocínio passo a passo longo; consolide apenas a justificativa final."
        if structured_reasoning
        else "Faça uma análise direta e concisa."
    )
    return (
        "Entrada congelada para diagnóstico de carteira:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        f"{reasoning_instruction}\n"
        "Produza o diagnóstico final conforme o schema."
    )
