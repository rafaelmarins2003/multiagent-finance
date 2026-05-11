import json
from typing import Any

from mafin.agents.base import Agent
from mafin.data.compact import compact_macro_data
from mafin.schema import MacroAnalysisOutput

SYSTEM_PROMPT = """Você é um analista macroeconômico para carteiras brasileiras.

Seu papel: avaliar exposições macroeconômicas da carteira a partir dos setores,
pesos e indicadores macro explicitamente fornecidos no input.

Restrições:
- Não recomende compra ou venda.
- Não invente indicadores macro, datas, inflação, juros, câmbio ou eventos.
- Diferencie implicações inferidas da composição setorial de lacunas de dados.
- Não use preço, média móvel, volatilidade ou tendência técnica como proxy macroeconômico.
"""


class MacroAgent(Agent):
    role = "macro"
    system_prompt = SYSTEM_PROMPT

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        portfolio = state["portfolio"]
        profile = state.get("profile", {})
        macro_data = state.get("macro_data", {})

        prompt = (
            f"Carteira (ticker, peso, setor):\n"
            f"{json.dumps(portfolio, ensure_ascii=False, indent=2)}\n\n"
            f"Perfil do usuário:\n{json.dumps(profile, ensure_ascii=False, indent=2)}\n\n"
            f"Indicadores macro disponíveis:\n"
            f"{json.dumps(compact_macro_data(macro_data), ensure_ascii=False, indent=2)}\n\n"
            "Produza a análise macroeconômica conforme o schema."
        )

        output = self._invoke_structured(prompt, schema=MacroAnalysisOutput)
        return {
            "analyses": [
                {
                    "role": self.role,
                    "model": self.model,
                    "output": output.model_dump(),
                }
            ]
        }
