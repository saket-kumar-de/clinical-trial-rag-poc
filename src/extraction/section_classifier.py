"""
Classifies chunks of protocol text into sections (objectives, eligibility
criteria, endpoints, etc.) using a rule-based first pass.

Interview talking point: this is deliberately rule-based for the POC —
the production path would swap in an LLM-based classifier, which is the
scaling move once section headers stop being this predictable.
"""
import re

SECTION_PATTERNS = {
    "objectives": re.compile(r"(?i)\bobjectives?\b"),
    "eligibility_criteria": re.compile(r"(?i)\beligibility criteria\b|\binclusion|exclusion criteria\b"),
    "endpoints": re.compile(r"(?i)\b(primary|secondary) endpoints?\b"),
    "study_design": re.compile(r"(?i)\bstudy design\b"),
}


def classify_section(heading_text: str) -> str | None:
    """
    Match a heading line against known section patterns.
    Returns the section_type string, or None if no pattern matches.
    TODO: extend with more patterns as real protocol PDFs reveal edge cases.
    """
    for section_type, pattern in SECTION_PATTERNS.items():
        if pattern.search(heading_text):
            return section_type
    return None
