"""
FastAPI serving layer — exposes the retrieval graph (and, later,
structured trial queries) over HTTP.
"""
from fastapi import FastAPI, HTTPException, Query

from src.retrieval.langgraph_node import build_retrieval_graph

app = FastAPI(title="Clinical Trial RAG POC")
_retrieval_graph = build_retrieval_graph()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/search")
def search(
    query: str = Query(..., min_length=1, description="Natural language question"),
    top_k: int = Query(5, ge=1, le=50, description="Number of results to return"),
) -> dict:
    """
    Run the retrieval graph for a natural-language question and return
    the matching chunks. min_length/ge/le are FastAPI's declarative
    validation -- an empty query or an out-of-range top_k is rejected
    automatically, before this function body ever runs.
    """
    try:
        final_state = _retrieval_graph.invoke(
            {
                "query": query,
                "top_k": top_k,
                "section_filter": None,
                "results": [],
                "no_match": False,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    return {
        "query": query,
        "section_filter": final_state["section_filter"],
        "no_match": final_state["no_match"],
        "results": final_state["results"],
    }