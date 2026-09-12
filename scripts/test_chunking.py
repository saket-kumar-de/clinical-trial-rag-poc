import sys
from pathlib import Path

sys.path.insert(0, ".")
from src.ingestion.pdf_loader import extract_text_and_headings
from src.extraction.section_classifier import classify_section

nct_ids = {
    "NCT03961204_protocol.pdf": "NCT03961204",
    "NCT04297917_protocol.pdf": "NCT04297917",
}

with open("chunk_boundaries.txt", "w", encoding="utf-8") as out:
    for pdf_path in sorted(Path("data/raw").glob("*.pdf")):
        text, headings = extract_text_and_headings(str(pdf_path))

        positions = []
        search_from = 0
        for heading in headings:
            pos = text.find(heading, search_from)
            if pos == -1:
                continue
            positions.append((pos, heading, classify_section(heading)))
            search_from = pos + len(heading)

        out.write(f"\n{'=' * 60}\n{pdf_path.name}\n{'=' * 60}\n")
        for i, (start, heading, section_type) in enumerate(positions):
            if not section_type:
                continue
            next_heading = positions[i + 1][1] if i + 1 < len(positions) else "(end of document)"
            next_type = positions[i + 1][2] if i + 1 < len(positions) else None
            out.write(f"\n[{section_type}] STARTS AT: {heading!r}\n")
            out.write(f"  ENDS BECAUSE NEXT HEADING IS: {next_heading!r} (typed as: {next_type!r})\n")

print("Written to chunk_boundaries.txt")