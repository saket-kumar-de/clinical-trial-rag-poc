"""
Splits extracted document text into chunks for embedding, using
heading positions to find section boundaries.

Chunking by section (not fixed token windows) is the deliberate choice
here: an eligibility criterion or endpoint description shouldn't be
split mid-sentence just because it crossed a token boundary.
"""
import re
from dataclasses import dataclass

from src.extraction.section_classifier import classify_section

# A section body longer than this gets split further, on sentence
# boundaries. Set well under all-MiniLM-L6-v2's real 256-token input
# limit, not just "a reasonable-looking size" -- at a conservative
# ~3.5 characters per token (clinical text with long drug/medical
# terms tokenizes denser than general English), 800 chars is ~228
# tokens, leaving real margin. The previous 1500-char value was
# ~330-430 tokens depending on the estimate -- well past the limit,
# meaning the back half of many real chunks would have been silently
# truncated by the model and never actually embedded.
_MAX_CHUNK_CHARS = 800

# A chunk shorter than this carries essentially no embeddable content
# -- confirmed on real PDF data: a genuine "chunk" was a 3-character
# section number ("4.1") left over when two headings sat very close
# together. Calibrated for PDF extraction noise specifically -- clean,
# already-correctly-scoped structured data (a ClinicalTrials.gov
# endpoint description, a PubMed abstract) can legitimately be shorter
# than this and still be real content, not noise. See chunk_text's
# min_length parameter.
_MIN_CHUNK_CHARS = 30

# Splits after '.', '!', or '?' followed by whitespace. A simple
# heuristic, not a real sentence tokenizer -- it will mis-split on
# abbreviations ("Dr.", "e.g."). Acceptable for this POC's scope; an
# NLP sentence splitter (spaCy, nltk) is the upgrade path if this ever
# needs to be robust across a large, varied real corpus.
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


@dataclass
class TextChunk:
    nct_id: str
    source_doc: str
    section_type: str
    text: str


def chunk_text(nct_id: str, source_doc: str, section_type: str, text: str, min_length: int = _MIN_CHUNK_CHARS) -> list[TextChunk]:
    """
    Split `text` to size (via _split_long_text) and wrap each surviving
    piece into a TextChunk. For content that's already correctly
    scoped to one section -- a ClinicalTrials.gov eligibility
    criterion, a PubMed abstract -- this is the whole job: no
    heading-boundary-finding needed, just split-to-size and package.

    chunk_by_section calls this internally once it has found each
    section's body text; the ClinicalTrials.gov/PubMed ingestion path
    (run_ingestion.py) calls it directly, since that text arrives
    pre-scoped to its section already.

    `min_length` defaults to _MIN_CHUNK_CHARS, tuned for PDF noise
    (see that constant's comment). Confirmed on real data this default
    is wrong for clean structured fields: real ClinicalTrials.gov
    endpoint descriptions like "Overall survival" (16 chars) or
    "Progression-free survival" (25 chars) are completely legitimate
    and shorter than 30 -- callers with already-clean, pre-scoped text
    should pass a much smaller min_length (e.g. 1, to still drop a
    genuinely empty string) rather than accept the PDF-tuned default.
    """
    chunks: list[TextChunk] = []
    for piece in _split_long_text(text):
        if len(piece) >= min_length:
            chunks.append(TextChunk(nct_id=nct_id, source_doc=source_doc, section_type=section_type, text=piece))
    return chunks


def chunk_by_section(nct_id: str, source_doc: str, full_text: str, headings: list[str]) -> list[TextChunk]:
    """
    Slice `full_text` into per-section chunks, using `headings` (the
    filtered output of extract_text_and_headings, in document order)
    to find where each section starts and ends.

    Every heading -- typed or not -- acts as a boundary, so a real
    section's body correctly stops at the next heading of any kind
    (e.g. an untyped "References" heading right after a typed
    section), rather than absorbing everything up to the next
    *typed* heading, however far away that is. Chunks are only
    emitted for headings classify_section() recognizes as one of the
    four target types; untyped headings still bound their neighbors
    but produce no chunk of their own.
    """
    positions: list[tuple[int, str, str | None]] = []
    search_from = 0
    for heading in headings:
        position = full_text.find(heading, search_from)
        if position == -1:
            continue  # extracted heading text not found from this point on -- skip rather than guess
        positions.append((position, heading, classify_section(heading)))
        search_from = position + len(heading)

    chunks: list[TextChunk] = []
    for i, (start, heading, section_type) in enumerate(positions):
        if not section_type:
            continue

        body_start = start + len(heading)
        body_end = positions[i + 1][0] if i + 1 < len(positions) else len(full_text)
        body = full_text[body_start:body_end].strip()

        chunks.extend(chunk_text(nct_id, source_doc, section_type, body))

    return chunks


def _split_long_text(text: str) -> list[str]:
    """
    Split `text` into pieces no longer than _MAX_CHUNK_CHARS,
    breaking only at sentence boundaries. Text already under the
    limit is returned as a single-element list, unchanged.
    """
    if len(text) <= _MAX_CHUNK_CHARS:
        return [text]

    sentences = _SENTENCE_SPLIT_PATTERN.split(text)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > _MAX_CHUNK_CHARS:
            pieces.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current.strip())
    return pieces