import json
from typing import Any

from mafin.agents.base import Agent
from mafin.schema import RiskAnalysisOutput

SYSTEM_PROMPT = """Você é um analista de risco de carteiras.

Seu papel: avaliar concentração, volatilidade informada, diversificação setorial e
alinhamento da carteira ao perfil do usuário.

Restrições:
- Não recomende compra ou venda.
- Use apenas pesos, setores, perfil e métricas de risco presentes no input.
- Se uma métrica necessária não estiver disponível, registre a lacuna.
- Não infira sobrevalorização ou subvalorização sem dados explícitos de valuation.
"""


class RiskAgent(Agent):
    role = "risk"
    system_prompt = SYSTEM_PROMPT

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        portfolio = state["portfolio"]
        profile = state.get("profile", {})
        market = state.get("market_data", {})

        prompt = (
            f"Carteira (ticker, peso, setor):\n"
            f"{json.dumps(portfolio, ensure_ascii=False, indent=2)}\n\n"
            f"Perfil do usuário:\n{json.dumps(profile, ensure_ascii=False, indent=2)}\n\n"
            f"Dados de mercado e risco por ticker:\n"
            f"{json.dumps(market, ensure_ascii=False, indent=2)}\n\n"
            "Produza a análise de risco conforme o schema."
        )

        output = self._invoke_structured(prompt, schema=RiskAnalysisOutput)
        return {
            "analyses": [
                {
                    "role": self.role,
                    "model": self.model,
                    "output": output.model_dump(),
                }
            ]
        }
