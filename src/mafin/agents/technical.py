import json
from typing import Any

from mafin.agents.base import Agent
from mafin.schema import TechnicalAnalysisOutput

SYSTEM_PROMPT = """Você é um analista técnico de mercado de ações brasileiro.

Seu papel: avaliar a situação técnica de cada ativo da carteira a partir dos dados
fornecidos (preços recentes, médias móveis, volatilidade, volume, indicadores).

Restrições:
- Não recomende compra ou venda. Apenas descreva o estado técnico.
- Cite números quando disponíveis no input. Não invente valores.
- Seja sucinto: 2-4 frases por ativo, no máximo.
- Não use linguagem de oportunidade, entrada, saída ou ajuste.
"""


class TechnicalAgent(Agent):
    role = "technical"
    system_prompt = SYSTEM_PROMPT

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        portfolio = state["portfolio"]
        market = state.get("market_data", {})

        prompt = (
            f"Carteira (ticker, peso, setor):\n"
            f"{json.dumps(portfolio, ensure_ascii=False, indent=2)}\n\n"
            f"Dados de mercado por ticker:\n{json.dumps(market, ensure_ascii=False, indent=2)}\n\n"
            "Produza a análise técnica conforme o schema."
        )

        output = self._invoke_structured(prompt, schema=TechnicalAnalysisOutput)
        return {
            "analyses": [
                {
                    "role": self.role,
                    "model": self.model,
                    "output": output.model_dump(),
                }
            ]
        }
