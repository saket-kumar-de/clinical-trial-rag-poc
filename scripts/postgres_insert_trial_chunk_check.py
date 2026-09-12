# import sys
# sys.path.insert(0, ".")
# from pathlib import Path
# from src.storage.db import insert_trial, insert_chunks
# from src.storage.models import Trial
# from src.ingestion.pdf_loader import extract_text_and_headings
# from src.embedding.chunker import chunk_by_section
# from src.embedding.embedder import embed_texts

# nct_ids = {"NCT03961204_protocol.pdf": "NCT03961204", "NCT04297917_protocol.pdf": "NCT04297917"}

# for pdf_path in sorted(Path("data/raw").glob("*.pdf")):
#     nct_id = nct_ids[pdf_path.name]

#     # Satisfy the foreign key first -- a minimal stub row is enough for now
#     insert_trial(Trial(nct_id=nct_id, title=f"Placeholder title for {nct_id}", source="pdf"))

#     text, headings = extract_text_and_headings(str(pdf_path))
#     chunks = chunk_by_section(nct_id, pdf_path.name, text, headings)
#     embeddings = embed_texts([c.text for c in chunks])

#     insert_chunks(chunks, embeddings)
#     print(f"{nct_id}: inserted {len(chunks)} chunks")

# print("Done.")

from src.storage.db import get_connection

conn = get_connection()
with conn.cursor() as cur:
    cur.execute("SELECT nct_id, section_type, COUNT(*) FROM chunks GROUP BY nct_id, section_type ORDER BY nct_id, section_type;")
    for row in cur.fetchall():
        print(row)
conn.close()