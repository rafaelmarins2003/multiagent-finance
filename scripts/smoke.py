"""Smoke test end-to-end com carteira hardcoded.

Pré-requisitos:
  1. Ollama rodando em localhost:11434
  2. Modelo baixado:  ollama pull granite4:tiny-h
  3. Pacote instalado:  pip install -e .

Uso:
  python scripts/smoke.py
"""

from __future__ import annotations

import json

from mafin.graph.build import build_graph


SAMPLE_PORTFOLIO = [
    {"ticker": "PETR4", "weight": 0.30, "sector": "Energia"},
    {"ticker": "VALE3", "weight": 0.25, "sector": "Mineração"},
    {"ticker": "ITUB4", "weight": 0.25, "sector": "Financeiro"},
    {"ticker": "WEGE3", "weight": 0.20, "sector": "Industrial"},
]

SAMPLE_PROFILE = {
    "risk_tolerance": "moderate",
    "horizon": "long",
    "objective": "valorização de capital com tolerância média a drawdown",
}

# Mock determinístico de dados de mercado — substituir por yfinance no próximo passo.
SAMPLE_MARKET = {
    "PETR4": {"price": 38.50, "ma50": 39.10, "ma200": 36.20, "vol_30d": 0.28, "trend": "lateral"},
    "VALE3": {"price": 62.10, "ma50": 65.40, "ma200": 68.90, "vol_30d": 0.31, "trend": "baixa"},
    "ITUB4": {"price": 33.80, "ma50": 32.10, "ma200": 30.50, "vol_30d": 0.18, "trend": "alta"},
    "WEGE3": {"price": 41.20, "ma50": 40.50, "ma200": 38.10, "vol_30d": 0.22, "trend": "alta"},
}


def main() -> None:
    graph = build_graph()

    initial_state = {
        "portfolio": SAMPLE_PORTFOLIO,
        "profile": SAMPLE_PROFILE,
        "market_data": SAMPLE_MARKET,
        "analyses": [],
        "debate_rounds": [],
        "diagnosis": None,
    }

    result = graph.invoke(initial_state)

    print("=" * 60)
    print("ANALYSES")
    print("=" * 60)
    for a in result.get("analyses", []):
        print(f"\n[{a['role']}  ({a['model']})]")
        print(a["content"])

    print("\n" + "=" * 60)
    print("FULL STATE (debug)")
    print("=" * 60)
    print(json.dumps(
        {k: v for k, v in result.items() if k != "market_data"},
        ensure_ascii=False,
        indent=2,
        default=str,
    ))


if __name__ == "__main__":
    main()
