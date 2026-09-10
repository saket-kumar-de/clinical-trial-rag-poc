"""
Wraps pgvector similarity search as a single LangGraph node.
This is intentionally minimal, one node, one job, to demonstrate real
contact with the framework rather than a multi-agent graph the POC
doesn't need.
"""
from typing import TypedDict

from langgraph.graph import END, StateGraph


class RetrievalState(TypedDict):
    query: str
    top_k: int
    results: list[dict]


def retrieve_node(state: RetrievalState) -> RetrievalState:
    """
    Embed the query, run a cosine-similarity search against
    chunks.embedding, and return the top_k matches with their metadata.

    TODO:
      - Embed state["query"] via embed_texts
      - SELECT ... ORDER BY embedding <=> %s LIMIT %s
      - Populate state["results"] with
        {chunk_text, section_type, nct_id, score}
    """
    raise NotImplementedError


def build_retrieval_graph():
    """Assemble the single-node graph and compile it for invocation."""
    graph = StateGraph(RetrievalState)
    graph.add_node("retrieve", retrieve_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", END)
    return graph.compile()
