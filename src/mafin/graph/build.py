from itertools import pairwise

from langgraph.graph import END, START, StateGraph

from mafin.agents.debate import DebateOrchestrator
from mafin.agents.fundamental import FundamentalAgent
from mafin.agents.macro import MacroAgent
from mafin.agents.moderator import ModeratorAgent
from mafin.agents.risk import RiskAgent
from mafin.agents.sentiment import SentimentAgent
from mafin.agents.technical import TechnicalAgent
from mafin.config import LLM_MAX_CONCURRENCY, ROUTES, ModelRoute
from mafin.graph.metrics import instrument_node, summarize_execution
from mafin.graph.state import GraphState


def build_graph(
    routes: ModelRoute = ROUTES,
    *,
    enable_debate: bool = True,
    variant: str | None = None,
    parallel_specialists: bool | None = None,
):
    """Grafo B4: especialistas independentes, debate opcional e síntese final.

    Quando `LLM_MAX_CONCURRENCY > 1`, os especialistas fazem fan-out/fan-in.
    Debate e moderação continuam sequenciais para preservar dependências.
    """
    g = StateGraph(GraphState)
    execution_variant = variant or (routes.preset if enable_debate else "b3")
    use_parallel_specialists = (
        LLM_MAX_CONCURRENCY > 1 if parallel_specialists is None else parallel_specialists
    )

    specialist_nodes = [
        ("technical", TechnicalAgent(routes.technical)),
        ("sentiment", SentimentAgent(routes.sentiment)),
        ("fundamental", FundamentalAgent(routes.fundamental)),
        ("macro", MacroAgent(routes.macro)),
        ("risk", RiskAgent(routes.risk)),
    ]
    debate = DebateOrchestrator(routes)
    moderator = ModeratorAgent(routes.moderator)

    def metadata_node(_state: GraphState) -> dict:
        return {
            "model_route": routes.as_dict(),
            "execution_variant": execution_variant,
        }

    g.add_node("metadata", metadata_node)
    for name, agent in specialist_nodes:
        g.add_node(name, instrument_node(name, agent))

    g.add_node("moderator", instrument_node("moderator", moderator))
    g.add_node("metrics", summarize_execution)

    g.add_edge(START, "metadata")

    if enable_debate:
        g.add_node("debate", instrument_node("debate", debate))
        specialist_join_node = "debate"
        g.add_edge("debate", "moderator")
    else:
        specialist_join_node = "moderator"

    specialist_names = [name for name, _ in specialist_nodes]
    if use_parallel_specialists:
        for name in specialist_names:
            g.add_edge("metadata", name)
        g.add_edge(specialist_names, specialist_join_node)
    else:
        first_node = specialist_names[0]
        g.add_edge("metadata", first_node)
        for previous_name, next_name in pairwise(specialist_names):
            g.add_edge(previous_name, next_name)
        g.add_edge(specialist_names[-1], specialist_join_node)

    g.add_edge("moderator", "metrics")
    g.add_edge("metrics", END)

    return g.compile()
