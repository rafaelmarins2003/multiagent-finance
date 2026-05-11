from __future__ import annotations

from typing import Any

from mafin.agents.bear import BearAgent
from mafin.agents.bull import BullAgent
from mafin.config import DEBATE, ROUTES, DebateConfig, ModelRoute
from mafin.debate.convergence import convergence_score
from mafin.schema import DebateStatusOutput


class DebateOrchestrator:
    role = "debate_orchestrator"

    def __init__(
        self,
        routes: ModelRoute = ROUTES,
        config: DebateConfig = DEBATE,
    ):
        self.config = config
        self.bull = BullAgent(routes.bull)
        self.bear = BearAgent(routes.bear)

    def reset_runtime_metrics(self) -> None:
        self.bull.reset_runtime_metrics()
        self.bear.reset_runtime_metrics()

    def consume_runtime_metrics(self) -> dict[str, Any]:
        bull_metrics = self.bull.consume_runtime_metrics()
        bear_metrics = self.bear.consume_runtime_metrics()
        calls_by_model: dict[str, int] = {}
        for metrics in (bull_metrics, bear_metrics):
            for model, calls in metrics["calls_by_model"].items():
                calls_by_model[model] = calls_by_model.get(model, 0) + calls

        return {
            "llm_calls": bull_metrics["llm_calls"] + bear_metrics["llm_calls"],
            "calls_by_model": calls_by_model,
        }

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        debate_rounds: list[dict[str, Any]] = []
        previous_round: dict[str, Any] | None = None
        last_score = 0.0
        stopped_by = "max_rounds"

        for round_number in range(1, self.config.max_rounds + 1):
            context_rounds = state.get("debate_rounds", []) + debate_rounds
            bull_output = self.bull.run_argument(
                portfolio=state["portfolio"],
                profile=state.get("profile", {}),
                analyses=state.get("analyses", []),
                previous_rounds=context_rounds,
            )
            partial_round = {
                "round": round_number,
                "bull": {
                    "role": self.bull.role,
                    "model": self.bull.model,
                    "output": bull_output.model_dump(),
                },
            }

            bear_output = self.bear.run_argument(
                portfolio=state["portfolio"],
                profile=state.get("profile", {}),
                analyses=state.get("analyses", []),
                previous_rounds=context_rounds + [partial_round],
            )
            current_round = {
                "round": round_number,
                "bull": partial_round["bull"],
                "bear": {
                    "role": self.bear.role,
                    "model": self.bear.model,
                    "output": bear_output.model_dump(),
                },
            }

            if previous_round is not None:
                last_score = convergence_score(previous_round, current_round)
            current_round["convergence_vs_previous"] = last_score if previous_round else None
            debate_rounds.append(current_round)

            if (
                round_number >= self.config.min_rounds
                and previous_round is not None
                and last_score >= self.config.convergence_threshold
            ):
                stopped_by = "convergence"
                break

            previous_round = current_round

        status = DebateStatusOutput(
            stopped_by=stopped_by,
            rounds_completed=len(debate_rounds),
            convergence_score=last_score,
            converged=stopped_by == "convergence",
            summary=(
                "Debate encerrado por convergência adaptativa."
                if stopped_by == "convergence"
                else "Debate encerrado ao atingir o número máximo de rodadas."
            ),
        )

        return {
            "debate_rounds": debate_rounds,
            "debate_status": status.model_dump(),
        }
