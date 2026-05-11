from itertools import pairwise

from langgraph.graph import END, START, StateGraph

from mafin.agents.debate import DebateOrchestrator
from mafin.agents.fundamental import FundamentalAgent
from mafin.agents.macro import MacroAgent
from mafin.agents.moderator import ModeratorAgent
from mafin.agents.risk import RiskAgent
from mafin.agents.sentiment import SentimentAgent
from mafin.agents.technical import TechnicalAgent
from mafin.config import ROUTES, ModelRoute
from mafin.graph.metrics import instrument_node, summarize_execution
from mafin.graph.state import GraphState


def build_graph(
    routes: ModelRoute = ROUTES,
    *,
    enable_debate: bool = True,
    variant: str | None = None,
):
    """Grafo B4: especialistas sequenciais, debate opcional e síntese final.

    A execução sequencial preserva compatibilidade com Ollama Cloud, que pode
    restringir chamadas paralelas. Latência não é o gargalo nesta fase.
    """
    g = StateGraph(GraphState)
    execution_variant = variant or (routes.preset if enable_debate else "b3")

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

    first_node = specialist_nodes[0][0]
    g.add_edge(START, "metadata")
    g.add_edge("metadata", first_node)

    for (previous_name, _), (next_name, _) in pairwise(specialist_nodes):
        g.add_edge(previous_name, next_name)

    last_node = specialist_nodes[-1][0]
    if enable_debate:
        g.add_node("debate", instrument_node("debate", debate))
        g.add_edge(last_node, "debate")
        g.add_edge("debate", "moderator")
    else:
        g.add_edge(last_node, "moderator")

    g.add_edge("moderator", "metrics")
    g.add_edge("metrics", END)

    return g.compile()
