"""
Extracts raw text from clinical trial protocol PDFs.
Uses PyMuPDF (fast) with pdfplumber as a fallback for tricky layouts.
"""
import fitz  # PyMuPDF


def extract_text(pdf_path: str) -> str:
    """
    Extract all text from a PDF, page by page.
    TODO:
      - Open with fitz, iterate pages, concatenate text
      - Fall back to pdfplumber for pages where fitz returns near-empty
        text (common with scanned/image-based sections)
    """
    raise NotImplementedError
