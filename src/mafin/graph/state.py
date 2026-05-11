from operator import add
from typing import Annotated, Any, TypedDict


class GraphState(TypedDict, total=False):
    """State compartilhado entre nós do grafo.

    Campos com Annotated[..., add] são acumulados (reducer = list concat),
    permitindo que múltiplos agentes em paralelo escrevam sem conflito.
    """

    portfolio: list[dict[str, Any]]  # [{ticker, weight, sector}]
    profile: dict[str, Any]
    market_data: dict[str, Any]
    fundamental_data: dict[str, Any]
    sentiment_data: dict[str, Any]
    macro_data: dict[str, Any]

    analyses: Annotated[list[dict[str, Any]], add]
    debate_rounds: Annotated[list[dict[str, Any]], add]
    debate_status: dict[str, Any]

    diagnosis: dict[str, Any] | None
    model_route: dict[str, Any]
    execution_variant: str
    execution_metrics: Annotated[list[dict[str, Any]], add]
    execution_summary: dict[str, Any]
