"""
FastAPI serving layer — exposes the retrieval graph (and, later,
structured trial queries) over HTTP.
"""
from fastapi import FastAPI, Query

from src.retrieval.langgraph_node import build_retrieval_graph

app = FastAPI(title="Clinical Trial RAG POC")
_retrieval_graph = build_retrieval_graph()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/search")
def search(query: str = Query(..., description="Natural language question"), top_k: int = 5) -> dict:
    """
    TODO: invoke _retrieval_graph with {"query": query, "top_k": top_k}
    and return the results list.
    """
    raise NotImplementedError
