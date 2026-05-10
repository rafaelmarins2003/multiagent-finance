import json
from typing import Any

from mafin.agents.base import Agent

SYSTEM_PROMPT = """Você é um analista técnico de mercado de ações brasileiro.

Seu papel: avaliar a situação técnica de cada ativo da carteira a partir dos dados
fornecidos (preços recentes, médias móveis, volatilidade, volume, indicadores).

Restrições:
- Não recomende compra ou venda. Apenas descreva o estado técnico.
- Cite números quando disponíveis no input. Não invente valores.
- Seja sucinto: 2-4 frases por ativo, no máximo.

Saída obrigatória em JSON com a estrutura:
{
  "per_ticker": [{"ticker": str, "summary": str, "signals": [str]}],
  "overall": str
}
"""


class TechnicalAgent(Agent):
    role = "technical"
    system_prompt = SYSTEM_PROMPT

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        portfolio = state["portfolio"]
        market = state.get("market_data", {})

        prompt = (
            f"Carteira (ticker, peso, setor):\n{json.dumps(portfolio, ensure_ascii=False, indent=2)}\n\n"
            f"Dados de mercado por ticker:\n{json.dumps(market, ensure_ascii=False, indent=2)}\n\n"
            "Produza a análise técnica conforme o schema."
        )

        content = self._invoke(prompt)
        return {
            "analyses": [
                {"role": self.role, "model": self.model, "content": content}
            ]
        }
