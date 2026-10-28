"""
src/agent/graph.py
Compiles the LangGraph state machine with optional checkpoint persistence.
"""

from typing import Any, Literal, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from src.agent.state import AgentState
from src.agent.nodes import (
    router_node,
    decompose_node,
    metadata_tool_node,
    search_tool_node,
    generate_node,
)


def route_decision(state: AgentState) -> Literal["metadata_tool", "decomposer"]:
    """Reads the schema-validated decision and routes accordingly."""
    return "metadata_tool" if state.get("current_query") == "metadata" else "decomposer"


def build_agent_graph(
    retriever: Any,
    metadata_profile: str,
    llm: Any,
    checkpointer: Optional[BaseCheckpointSaver] = None
):
    """Builds and compiles the tool-enabled agent workflow with persistent memory."""
    workflow = StateGraph(AgentState)

    # 1. Register Nodes
    workflow.add_node("router", lambda state: router_node(state, llm))
    workflow.add_node("decomposer", lambda state: decompose_node(state, llm))
    workflow.add_node("metadata_tool", lambda state: metadata_tool_node(state, metadata_profile))
    workflow.add_node("search_tool", lambda state: search_tool_node(state, retriever))
    workflow.add_node("generate", lambda state: generate_node(state, llm))

    # 2. Add Conditional Routing
    workflow.add_edge(START, "router")
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "metadata_tool": "metadata_tool",
            "decomposer": "decomposer",
        },
    )

    # 3. Transitions
    workflow.add_edge("decomposer", "search_tool")
    workflow.add_edge("metadata_tool", "generate")
    workflow.add_edge("search_tool", "generate")
    workflow.add_edge("generate", END)

    # Compile with or without persistent checkpoint saver
    return workflow.compile(checkpointer=checkpointer)