import sys
from pathlib import Path

sys.path.insert(0, ".")
from src.ingestion.pdf_loader import extract_text_and_headings
from src.extraction.section_classifier import classify_section

for pdf_path in sorted(Path("data/raw").glob("*.pdf")):
    print(f"\n{'=' * 60}\n{pdf_path.name}\n{'=' * 60}")

    text, headings = extract_text_and_headings(str(pdf_path))
    print(f"Total text length: {len(text)}, heading candidates: {len(headings)}")

    typed = 0
    for h in headings:
        section_type = classify_section(h)
        if section_type:
            typed += 1
        print(f"  [{section_type or '-'}] {h!r}")

    print(f"\n{typed}/{len(headings)} heading candidates matched a known section type")