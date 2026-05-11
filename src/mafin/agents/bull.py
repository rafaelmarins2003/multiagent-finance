import json
from typing import Any

from mafin.agents.base import Agent
from mafin.schema import DebateArgumentOutput

SYSTEM_PROMPT = """Você é o agente Bull em um debate adversarial sobre diagnóstico de carteira.

Seu papel: construir a leitura favorável mais forte possível sobre a carteira em
relação ao perfil do usuário, usando apenas dados e análises fornecidos.

Restrições:
- Não recomende compra, venda ou rebalanceamento.
- Não ignore riscos; explique por que eles podem ser toleráveis ou mitigados.
- Não invente dados, eventos ou fundamentos.
- Seja direto e auditável.
"""


class BullAgent(Agent):
    role = "bull"
    system_prompt = SYSTEM_PROMPT

    def run_argument(
        self,
        *,
        portfolio: list[dict[str, Any]],
        profile: dict[str, Any],
        analyses: list[dict[str, Any]],
        previous_rounds: list[dict[str, Any]],
    ) -> DebateArgumentOutput:
        prompt = (
            f"Carteira:\n{json.dumps(portfolio, ensure_ascii=False, indent=2)}\n\n"
            f"Perfil:\n{json.dumps(profile, ensure_ascii=False, indent=2)}\n\n"
            f"Análises especializadas:\n{json.dumps(analyses, ensure_ascii=False, indent=2)}\n\n"
            f"Rodadas anteriores:\n{json.dumps(previous_rounds, ensure_ascii=False, indent=2)}\n\n"
            "Produza o argumento Bull conforme o schema."
        )
        output = self._invoke_structured(prompt, schema=DebateArgumentOutput)
        output.stance = "bull"
        return output

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        output = self.run_argument(
            portfolio=state["portfolio"],
            profile=state.get("profile", {}),
            analyses=state.get("analyses", []),
            previous_rounds=state.get("debate_rounds", []),
        )
        return {
            "debate_rounds": [
                {
                    "role": self.role,
                    "model": self.model,
                    "output": output.model_dump(),
                }
            ]
        }
