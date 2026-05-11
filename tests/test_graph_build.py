from mafin.config import ROUTES
from mafin.graph.build import build_graph


def test_build_graph_with_debate_compiles():
    graph = build_graph(ROUTES, enable_debate=True)
    nodes = set(graph.get_graph().nodes.keys())
    expected = {"technical", "sentiment", "fundamental", "macro", "risk", "debate", "moderator"}
    assert expected.issubset(nodes)


def test_build_graph_without_debate_compiles():
    graph = build_graph(ROUTES, enable_debate=False)
    nodes = set(graph.get_graph().nodes.keys())
    assert "debate" not in nodes
    assert {"technical", "sentiment", "fundamental", "macro", "risk", "moderator"}.issubset(nodes)


def test_build_graph_default_enables_debate():
    graph = build_graph()
    nodes = set(graph.get_graph().nodes.keys())
    assert "debate" in nodes
