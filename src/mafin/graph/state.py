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

    analyses: Annotated[list[dict[str, Any]], add]
    debate_rounds: Annotated[list[dict[str, Any]], add]

    diagnosis: dict[str, Any] | None
