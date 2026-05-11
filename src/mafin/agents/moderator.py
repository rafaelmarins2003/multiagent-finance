import json
from typing import Any

from mafin.agents.base import Agent
from mafin.schema import PortfolioDiagnosisOutput

SYSTEM_PROMPT = """Você é o moderador de um sistema multiagente de diagnóstico de carteiras.

Seu papel: consolidar análises especializadas em um diagnóstico final alinhado ao
perfil do usuário.

Classificações permitidas:
- estavel: carteira compatível com o perfil e sem riscos relevantes aparentes.
- atencao: há pontos de atenção, mas sem desalinhamento dominante.
- risco_elevado: riscos relevantes são dominantes para o perfil informado.
- desalinhada: carteira incompatível com tolerância, horizonte ou objetivo do usuário.
- inconclusiva: dados insuficientes para diagnóstico confiável.

Restrições:
- Não recomende compra, venda ou rebalanceamento específico.
- Não sugira ajustes futuros específicos; limite-se ao diagnóstico.
- Use apenas as análises e dados fornecidos.
- Deixe claro quando a confiança for limitada por lacunas de dados.
- Não transforme lacunas de dados em risco financeiro; se elas dominarem, use inconclusiva.
- Seja conciso e não use Markdown na justificativa.
"""


class ModeratorAgent(Agent):
    role = "moderator"
    system_prompt = SYSTEM_PROMPT

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        portfolio = state["portfolio"]
        profile = state.get("profile", {})
        analyses = state.get("analyses", [])
        debate_rounds = state.get("debate_rounds", [])
        debate_status = state.get("debate_status", {})

        prompt = (
            f"Carteira:\n{json.dumps(portfolio, ensure_ascii=False, indent=2)}\n\n"
            f"Perfil do usuário:\n{json.dumps(profile, ensure_ascii=False, indent=2)}\n\n"
            f"Análises especializadas:\n{json.dumps(analyses, ensure_ascii=False, indent=2)}\n\n"
            f"Rodadas de debate Bull/Bear:\n"
            f"{json.dumps(debate_rounds, ensure_ascii=False, indent=2)}\n\n"
            f"Status do debate:\n{json.dumps(debate_status, ensure_ascii=False, indent=2)}\n\n"
            "Produza o diagnóstico final conforme o schema."
        )

        output = self._invoke_structured(prompt, schema=PortfolioDiagnosisOutput)
        return {
            "diagnosis": {
                "role": self.role,
                "model": self.model,
                "output": output.model_dump(),
            }
        }
