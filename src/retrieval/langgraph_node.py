"""
Wraps pgvector similarity search as a small LangGraph pipeline:
classify the query -> retrieve -> deduplicate near-identical results ->
route on confidence. Deliberately stops short of a generation step
(no LLM call) -- retrieval quality, not answer synthesis, is the part
that matches this project's actual scope.
"""
from typing import TypedDict

import numpy as np
from langgraph.graph import END, StateGraph

from src.embedding.embedder import embed_texts
from src.storage.db import get_connection

# Keyword -> section_type routing for classify_query_node. A cheap,
# instant keyword check, not an LLM call -- this runs on every query,
# so it needs to be free and immediate.
_SECTION_KEYWORDS = {
    "eligibility_criteria": ["eligib", "inclusion", "exclusion", "criteria"],
    "endpoints": ["endpoint", "outcome"],
    "study_design": ["design", "methodology", "arm", "randomiz"],
    "objectives": ["objective", "aim", "purpose"],
}

# Two results count as near-duplicates above this cosine similarity.
# A starting guess, not a calibrated value -- like every other
# threshold in this project (_MIN_CHUNK_CHARS, _HEADING_SIZE_RATIO,
# etc.), this needs tuning against real query results, not trusted on
# first guess.
_DUPLICATE_SIMILARITY_THRESHOLD = 0.95

# Below this top-result score, treat the query as having no real
# match rather than returning weak, likely-irrelevant chunks anyway.
# Started at 0.3; lowered after real testing showed a genuinely
# correct endpoint match scoring 0.317 -- uncomfortably close to the
# old cutoff. Endpoint-type matches score consistently lower than
# eligibility_criteria matches on real data (likely denser technical
# abbreviations embedding less cleanly) -- a single threshold across
# both is a compromise, not a fully calibrated value; a false
# negative here (rejecting a real match) looks broken to a user in a
# way a slightly-weak accepted match doesn't, so this errs toward
# permissive until there's enough real data to justify a
# per-section_type threshold instead of one shared guess.
_CONFIDENCE_THRESHOLD = 0.25


class RetrievalState(TypedDict):
    query: str
    top_k: int
    section_filter: str | None
    results: list[dict]
    no_match: bool


def classify_query_node(state: RetrievalState) -> RetrievalState:
    """
    Cheap keyword check to decide whether this query is clearly about
    one specific section type -- if so, retrieve_node narrows its
    search to just that type instead of searching everything.
    """
    query_lower = state["query"].lower()
    state["section_filter"] = None
    for section_type, keywords in _SECTION_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            state["section_filter"] = section_type
            break
    return state


def retrieve_node(state: RetrievalState) -> RetrievalState:
    """
    Embed the query, run a cosine-similarity search against
    chunks.embedding, and return the top_k matches with their metadata.

    `embedding <=> %s` is pgvector's cosine-DISTANCE operator (lower =
    more similar) -- the same operator the HNSW index was built with
    (vector_cosine_ops), so this query actually uses the index rather
    than falling back to a sequential scan. Converted to a similarity
    SCORE (1 - distance) before returning, matching how similarity has
    been discussed throughout this project (higher = more similar).

    Also returns each row's raw embedding -- needed by
    deduplicate_node, stripped back out before the graph finishes.
    """
    query_vector = embed_texts([state["query"]])[0]

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            if state.get("section_filter"):
                cur.execute(
                    """
                    SELECT chunk_text, section_type, nct_id, embedding,
                           embedding <=> %s::vector AS distance
                    FROM chunks
                    WHERE section_type = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_vector, state["section_filter"], query_vector, state["top_k"]),
                )
            else:
                cur.execute(
                    """
                    SELECT chunk_text, section_type, nct_id, embedding,
                           embedding <=> %s::vector AS distance
                    FROM chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_vector, query_vector, state["top_k"]),
                )
            rows = cur.fetchall()
    finally:
        conn.close()

    state["results"] = [
        {
            "chunk_text": row["chunk_text"],
            "section_type": row["section_type"],
            "nct_id": row["nct_id"],
            "embedding": row["embedding"].to_numpy(),
            "score": 1 - row["distance"],
        }
        for row in rows
    ]
    return state


def deduplicate_node(state: RetrievalState) -> RetrievalState:
    """
    Drop chunks whose embedding is highly similar to one already kept
    -- targets near-duplicate restatements (confirmed real pattern:
    NCT04297917 restates eligibility criteria across amendment
    history), not a blunt per-trial count cap. Embeddings are already
    normalized (embed_texts uses normalize_embeddings=True), so a
    plain dot product IS the cosine similarity -- no need to divide
    by vector norms.
    """
    kept = []
    for r in state["results"]:
        is_near_duplicate = any(
            np.dot(r["embedding"], k["embedding"]) > _DUPLICATE_SIMILARITY_THRESHOLD for k in kept
        )
        if not is_near_duplicate:
            kept.append(r)

    # Strip the embedding back out -- it served its purpose here;
    # nothing downstream (no_match_node, the eventual API response)
    # needs a 384-number vector riding along with each result.
    state["results"] = [{k: v for k, v in r.items() if k != "embedding"} for r in kept]
    return state


def filter_low_confidence_node(state: RetrievalState) -> RetrievalState:
    """
    Drop individual results below _CONFIDENCE_THRESHOLD, rather than
    keeping or discarding the whole top-k batch based only on the
    single best result's score. Without this, a query where result 1
    scores well but results 3-5 score poorly would return all 5 as if
    equally trustworthy, since nothing ever inspected anything past
    the first one.
    """
    state["results"] = [r for r in state["results"] if r["score"] >= _CONFIDENCE_THRESHOLD]
    return state


def check_confidence(state: RetrievalState) -> str:
    """
    Routing function (not a node) for the conditional edge after
    filter_low_confidence_node: by this point, every remaining result
    has already individually cleared the threshold, so this only
    needs to check whether anything survived at all.
    """
    return "good" if state["results"] else "no_match"


def no_match_node(state: RetrievalState) -> RetrievalState:
    """
    Low-confidence path: clear results and flag no_match explicitly,
    rather than silently returning weak, likely-irrelevant chunks.
    """
    state["results"] = []
    state["no_match"] = True
    return state


def build_retrieval_graph():
    """
    classify_query -> retrieve -> deduplicate -> filter_low_confidence
                                                        |
                                              [confidence branch]
                                                /            \
                                             (good)       (empty)
                                               END      no_match -> END
    """
    graph = StateGraph(RetrievalState)
    graph.add_node("classify_query", classify_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("deduplicate", deduplicate_node)
    graph.add_node("filter_low_confidence", filter_low_confidence_node)
    graph.add_node("no_match", no_match_node)

    graph.set_entry_point("classify_query")
    graph.add_edge("classify_query", "retrieve")
    graph.add_edge("retrieve", "deduplicate")
    graph.add_edge("deduplicate", "filter_low_confidence")
    graph.add_conditional_edges("filter_low_confidence", check_confidence, {"good": END, "no_match": "no_match"})
    graph.add_edge("no_match", END)
    return graph.compile()