import json
from typing import Any

from mafin.agents.base import Agent
from mafin.schema import DebateArgumentOutput

SYSTEM_PROMPT = """Você é o agente Bear em um debate adversarial sobre diagnóstico de carteira.

Seu papel: construir a leitura crítica mais forte possível sobre a carteira em
relação ao perfil do usuário, usando apenas dados e análises fornecidos.

Restrições:
- Não recomende compra, venda ou rebalanceamento.
- Não exagere riscos sem evidência explícita.
- Não transforme lacunas de dados em perdas financeiras; trate lacunas como incerteza.
- Não invente dados, eventos ou fundamentos.
- Seja direto e auditável.
- Seja conciso: tese com até 2 frases; listas com no máximo 3 itens; cada item curto.
"""


class BearAgent(Agent):
    role = "bear"
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
            "Produza o argumento Bear conforme o schema. Use no máximo 3 itens em "
            "`key_points`, `challenged_assumptions`, `evidence_used` e "
            "`residual_uncertainties`."
        )
        output = self._invoke_structured(prompt, schema=DebateArgumentOutput)
        output.stance = "bear"
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
