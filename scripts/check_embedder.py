import sys
from pathlib import Path

sys.path.insert(0, ".")
from src.ingestion.pdf_loader import extract_text_and_headings
from src.embedding.chunker import chunk_by_section
from src.embedding.embedder import embed_texts
import numpy as np

nct_ids = {
    "NCT03961204_protocol.pdf": "NCT03961204",
    "NCT04297917_protocol.pdf": "NCT04297917",
}

pdf_path = sorted(Path("data/raw").glob("*.pdf"))[0]
nct_id = nct_ids.get(pdf_path.name, "UNKNOWN")

print(f"Extracting and chunking {pdf_path.name}...")
text, headings = extract_text_and_headings(str(pdf_path))
chunks = chunk_by_section(nct_id, pdf_path.name, text, headings)
print(f"Got {len(chunks)} chunks\n")

print("Embedding all chunks (first run downloads the model, ~80MB, one-time)...")
texts = [c.text for c in chunks]
vectors = embed_texts(texts)

print(f"\nNumber of vectors: {len(vectors)}")
print(f"Dimensions per vector: {len(vectors[0])} (should be 384)")
print(f"Vector length (should be ~1.0, confirming normalization): {np.linalg.norm(vectors[0]):.4f}")

# The real test: does a semantic query actually retrieve the right chunk types?
query = "What are the eligibility criteria for participants?"
query_vector = embed_texts([query])[0]

print(f"\nQuery: {query!r}")
print("\nTop 5 most similar chunks (should be dominated by eligibility_criteria):")
scored = [(np.dot(query_vector, v), c.section_type, c.text[:80]) for v, c in zip(vectors, chunks)]
scored.sort(reverse=True)
for sim, section_type, preview in scored[:5]:
    print(f"  {sim:.3f} [{section_type}] {preview!r}...")