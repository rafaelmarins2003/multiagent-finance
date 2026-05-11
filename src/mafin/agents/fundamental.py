import json
from typing import Any

from mafin.agents.base import Agent
from mafin.data.compact import compact_fundamental_data
from mafin.schema import FundamentalAnalysisOutput

SYSTEM_PROMPT = """Você é um analista fundamentalista de ações brasileiras.

Seu papel: avaliar fundamentos por ativo apenas quando houver dados explícitos,
como receita, lucro, margens, dívida, valuation, dividendos ou indicadores similares.

Restrições:
- Não recomende compra ou venda.
- Não use conhecimento externo implícito sobre empresas.
- Se dados fundamentalistas não forem fornecidos, registre a lacuna em vez de inferir.
"""


class FundamentalAgent(Agent):
    role = "fundamental"
    system_prompt = SYSTEM_PROMPT

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        portfolio = state["portfolio"]
        fundamentals = state.get("fundamental_data", {})
        compacted_fundamentals = json.dumps(
            compact_fundamental_data(fundamentals),
            ensure_ascii=False,
            indent=2,
        )

        prompt = (
            f"Carteira (ticker, peso, setor):\n"
            f"{json.dumps(portfolio, ensure_ascii=False, indent=2)}\n\n"
            f"Dados fundamentalistas disponíveis por ticker:\n"
            f"{compacted_fundamentals}\n\n"
            "Produza a análise fundamentalista conforme o schema."
        )

        output = self._invoke_structured(prompt, schema=FundamentalAnalysisOutput)
        return {
            "analyses": [
                {
                    "role": self.role,
                    "model": self.model,
                    "output": output.model_dump(),
                }
            ]
        }
