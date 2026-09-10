"""
Splits classified text sections into chunks for embedding.

Chunking by section (not fixed token windows) is the deliberate choice
here: an eligibility criterion or endpoint description shouldn't be
split mid-sentence just because it crossed a token boundary.
"""
from dataclasses import dataclass


@dataclass
class TextChunk:
    nct_id: str
    source_doc: str
    section_type: str
    text: str


def chunk_by_section(nct_id: str, source_doc: str, sections: dict[str, str]) -> list[TextChunk]:
    """
    `sections` is {section_type: full_text_of_that_section}, already
    produced by the section classifier.

    TODO:
      - For short sections (e.g. a single eligibility criterion), keep
        as one chunk.
      - For long sections (e.g. a full objectives paragraph), split
        further on sentence boundaries with a max chunk length, but
        never split a section across two different section_types.
    """
    raise NotImplementedError
