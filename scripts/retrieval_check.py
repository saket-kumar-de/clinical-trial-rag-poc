import sys
sys.path.insert(0, ".")
from src.retrieval.langgraph_node import build_retrieval_graph

graph = build_retrieval_graph()

queries = [
    "What are the eligibility criteria for melanoma patients?",
    "What is the primary endpoint being measured?",
    "Tell me about the treatment approach for solid tumors",
    "What is the weather like today",
]

for query in queries:
    result = graph.invoke({"query": query, "top_k": 10, "section_filter": None, "results": [], "no_match": False})
    print(f"\nQuery: {query!r}")
    print(f"  section_filter used: {result['section_filter']}")
    print(f"  no_match: {result['no_match']}")
    print(f"  results returned: {len(result['results'])} (out of top_k=10 requested)")
    for r in result["results"]:
        print(f"    {r['score']:.3f} [{r['section_type']}] {r['nct_id']}: {r['chunk_text'][:80]!r}")