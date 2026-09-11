import sys
from pathlib import Path

sys.path.insert(0, ".")  # run this from the project root
import pymupdf as fitz
from src.ingestion.pdf_loader import (
    extract_text,
    _extract_with_pdfplumber,
    _looks_garbled,
    _vowel_ratio,
    _MIN_CHARS_PER_PAGE,
)

for pdf_path in sorted(Path("data/raw").glob("*.pdf")):
    print(f"\n{'=' * 60}\n{pdf_path.name}\n{'=' * 60}")

    # Diagnostic pass: report WHY each page did or didn't need the
    # fallback, without re-deciding what extract_text() already decides.
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    thin_pages = []
    garbled_pages = []
    fixed_pages = []
    still_garbled_pages = []

    for page_num, page in enumerate(doc):
        text = page.get_text().strip()
        is_thin = len(text) < _MIN_CHARS_PER_PAGE
        garbled = _looks_garbled(text)

        if is_thin:
            thin_pages.append(page_num)

        if garbled:
            ratio = _vowel_ratio(text)
            garbled_pages.append(page_num)
            if not is_thin:
                fallback = _extract_with_pdfplumber(str(pdf_path), page_num)
                if fallback and not _looks_garbled(fallback):
                    fixed_pages.append(page_num)
                    print(f"  page {page_num}: garbled (vowel_ratio={ratio:.3f}) -> FIXED by pdfplumber")
                else:
                    still_garbled_pages.append(page_num)
                    print(f"  page {page_num}: garbled (vowel_ratio={ratio:.3f}) -> still garbled, pdfplumber no better")
    doc.close()

    # Authoritative extraction — the real function, not a reimplementation.
    full_text = extract_text(str(pdf_path))

    print(f"\nTotal pages: {total_pages}")
    print(f"Total extracted chars: {len(full_text)}")
    print(f"Thin pages (0-indexed): {thin_pages}")
    print(f"Garbled pages detected (0-indexed): {garbled_pages}")
    if garbled_pages:
        print(f"  -> fixed by pdfplumber: {fixed_pages}")
        print(f"  -> still garbled after fallback: {still_garbled_pages}")
    print(f"\n--- First 500 chars ---\n{full_text[:500]}")
    print(f"\n--- Last 300 chars ---\n{full_text[-300:]}")