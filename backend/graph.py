"""
Builds and compiles the LangGraph StateGraph.
Compiled graph is cached via lru_cache so it is built only once.
"""
from functools import lru_cache

from langgraph.graph import StateGraph, END

from backend.state import AgentState
from backend.nodes import decision_node, tool_node, response_node


def route_after_decision(state: AgentState) -> str:
    d = state["decision"]
    if d.get("use_resume") or d.get("use_github") or d.get("use_leetcode"):
        return "tools"
    return "response"


@lru_cache
def get_graph():
    graph = StateGraph(AgentState)

    graph.add_node("decision", decision_node)
    graph.add_node("tools", tool_node)
    graph.add_node("response", response_node)

    graph.set_entry_point("decision")

    graph.add_conditional_edges("decision", route_after_decision, {
        "tools": "tools",
        "response": "response",
    })
    graph.add_edge("tools", "response")
    graph.add_edge("response", END)

    app = graph.compile()
    print("✅ LangGraph compiled")
    return app
