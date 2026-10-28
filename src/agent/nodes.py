"""
This file, contains the execution workers (nodes) for your LangGraph pipeline.

In LangGraph:

-Each node is a function that receives the current state.

-The node performs an action (calls a database, an LLM, or a parser).

-The node returns a standard Python dictionary containing only the fields it wants to update in AgentState. LangGraph automatically merges these returned values into the central state.
"""

from typing import Any, Dict, List
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from src.agent.state import AgentState, RouteDecision, DecompositionPlan
from src.agent.prompts import RAG_GENERATE_PROMPT


ROUTER_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert query router for an academic document agent.
Analyze the user's question and select the proper tool schema:

- 'metadata': User asks about the document's identity (author, student, title, university, supervisor, year, chapters).
- 'search': User asks about technical content, algorithms, equations, methodologies, comparisons, or experimental findings.

Question: {question}"""
)

DECOMPOSE_PROMPT = ChatPromptTemplate.from_template(
    """You are an academic query planner.
Break down the research question into 1 to 3 distinct search sub-queries optimized for retrieving excerpts from a thesis.
If the question is focused on a single topic, return exactly 1 sub-query.
If the question compares concepts across chapters or topics, break it into 2 or 3 separate, self-contained search queries.

Question: {question}"""
)


def router_node(state: AgentState, llm: Any) -> Dict[str, Any]:
    """Routes queries with strict Pydantic schema validation."""
    question = state["question"]
    structured_router = llm.with_structured_output(RouteDecision)
    router_chain = ROUTER_PROMPT | structured_router

    try:
        result: RouteDecision = router_chain.invoke({"question": question})
        return {
            "current_query": result.decision,
            "routing_reason": result.reasoning
        }
    except Exception as e:
        # Graceful fallback in the event of local model formatting hiccup
        return {
            "current_query": "search",
            "routing_reason": f"Fallback triggered due to schema parsing error: {e}"
        }


def decompose_node(state: AgentState, llm: Any) -> Dict[str, Any]:
    """Decomposes multi-part queries using structured Pydantic output."""
    question = state["question"]
    structured_decomposer = llm.with_structured_output(DecompositionPlan)
    decompose_chain = DECOMPOSE_PROMPT | structured_decomposer

    try:
        plan: DecompositionPlan = decompose_chain.invoke({"question": question})
        queries = plan.sub_queries if plan.sub_queries else [question]
    except Exception:
        # Fallback to the unaltered question if parsing encounters an edge case
        queries = [question]

    return {"sub_queries": queries}


def metadata_tool_node(state: AgentState, metadata_profile: str) -> Dict[str, Any]:
    """Retrieves authoritative front-matter metadata directly without search loss."""
    doc = Document(
        page_content=metadata_profile,
        metadata={"chapter": "Front Matter / Metadata", "page_number": 1}
    )
    return {"documents": [doc]}


def search_tool_node(state: AgentState, retriever: Any) -> Dict[str, Any]:
    """Executes multi-query hybrid retrieval and deduplicates chunks."""
    sub_queries = state.get("sub_queries", [state["question"]])

    seen_texts = set()
    combined_docs: List[Document] = []

    for sq in sub_queries:
        docs = retriever.invoke(sq)
        for doc in docs:
            if doc.page_content not in seen_texts:
                seen_texts.add(doc.page_content)
                combined_docs.append(doc)

    return {"documents": combined_docs}


def generate_node(state: AgentState, llm: Any) -> Dict[str, Any]:
    """Synthesizes the final answer using retrieved context."""
    question = state["question"]
    docs = state.get("documents", [])

    context_str = (
        "\n\n---\n\n".join(doc.page_content for doc in docs)
        if docs
        else "No relevant context found in document."
    )

    generator_chain = RAG_GENERATE_PROMPT | llm | StrOutputParser()
    answer = generator_chain.invoke({
        "question": question,
        "context": context_str
    })

    return {"generation": answer}