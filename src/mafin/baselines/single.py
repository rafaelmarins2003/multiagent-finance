from __future__ import annotations

from collections import Counter
from typing import Any

from mafin.agents.base import Agent
from mafin.baselines.prompts import render_diagnosis_prompt
from mafin.config import DEFAULT_LLM, DEFAULT_TEMPERATURE
from mafin.schema import DiagnosisClassification, PortfolioDiagnosisOutput

SYSTEM_PROMPT = """Você é uma LLM única usada como baseline de diagnóstico de carteira.

Seu papel é produzir diagnóstico personalizado de uma carteira financeira em
relação ao perfil do usuário, usando apenas os dados fornecidos.

Restrições:
- Não recomende compra, venda ou rebalanceamento específico.
- Não invente dados ausentes.
- Não trate lacunas de dados como perdas financeiras.
- Se os dados forem insuficientes, reduza a confiança ou classifique como inconclusiva.
- Use a mesma estrutura de saída do sistema multiagente.
"""


class SingleShotDiagnosisAgent(Agent):
    role = "single_llm"
    system_prompt = SYSTEM_PROMPT

    def __init__(
        self,
        model: str = DEFAULT_LLM,
        *,
        structured_reasoning: bool = False,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        super().__init__(model=model, temperature=temperature)
        self.structured_reasoning = structured_reasoning

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        output = self.run_diagnosis(state)
        return {
            "diagnosis": {
                "role": self.role,
                "model": self.model,
                "output": output.model_dump(),
            }
        }

    def run_diagnosis(self, state: dict[str, Any]) -> PortfolioDiagnosisOutput:
        prompt = render_diagnosis_prompt(
            state,
            structured_reasoning=self.structured_reasoning,
        )
        return self._invoke_structured(prompt, schema=PortfolioDiagnosisOutput)


def _unique(items: list[str], *, limit: int = 10) -> list[str]:
    seen = set()
    output = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized.lower() in seen:
            continue
        seen.add(normalized.lower())
        output.append(normalized)
        if len(output) >= limit:
            break
    return output


def aggregate_self_consistency(
    samples: list[PortfolioDiagnosisOutput],
) -> PortfolioDiagnosisOutput:
    if not samples:
        raise ValueError("At least one sample is required for self-consistency.")

    counts = Counter(sample.classification for sample in samples)
    avg_confidence_by_class = {
        classification: sum(
            sample.confidence for sample in samples if sample.classification == classification
        )
        / count
        for classification, count in counts.items()
    }
    winning_class = sorted(
        counts.keys(),
        key=lambda classification: (
            counts[classification],
            avg_confidence_by_class[classification],
            classification,
        ),
        reverse=True,
    )[0]
    winning_samples = [sample for sample in samples if sample.classification == winning_class]
    representative = sorted(winning_samples, key=lambda sample: sample.confidence, reverse=True)[0]

    positive_factors = _unique(
        [factor for sample in winning_samples for factor in sample.positive_factors]
    )
    negative_factors = _unique(
        [factor for sample in winning_samples for factor in sample.negative_factors]
    )
    mean_confidence = sum(sample.confidence for sample in winning_samples) / len(winning_samples)

    classification: DiagnosisClassification = winning_class
    return PortfolioDiagnosisOutput(
        classification=classification,
        justification=(
            f"Diagnóstico agregado por self-consistency: {counts[winning_class]}/"
            f"{len(samples)} amostras classificaram como {winning_class}. "
            f"Justificativa representativa: {representative.justification}"
        ),
        positive_factors=positive_factors,
        negative_factors=negative_factors,
        confidence=round(mean_confidence, 4),
        profile_alignment=representative.profile_alignment,
    )
