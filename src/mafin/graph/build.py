from langgraph.graph import END, START, StateGraph

from mafin.agents.technical import TechnicalAgent
from mafin.config import ROUTES, ModelRoute
from mafin.graph.state import GraphState


def build_graph(routes: ModelRoute = ROUTES):
    """Grafo V1: apenas o agente Técnico, fim a fim.

    Próximos incrementos:
      - adicionar Sentiment, Fundamental, Macro, Risk em paralelo (fan-out a partir de START)
      - rodada de debate Bull/Bear com terminação adaptativa (conditional edges)
      - Moderator como nó de síntese final
    """
    g = StateGraph(GraphState)

    technical = TechnicalAgent(routes.technical)
    g.add_node("technical", technical.run)

    g.add_edge(START, "technical")
    g.add_edge("technical", END)

    return g.compile()
