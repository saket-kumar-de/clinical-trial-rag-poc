"""
Extracts raw text from clinical trial protocol PDFs.
Uses PyMuPDF (fast) with pdfplumber as a fallback for tricky layouts.
"""
import re
from collections import Counter, defaultdict

import pymupdf as fitz  # `import fitz` still works but is deprecated as of PyMuPDF 1.28

# Below this character count, a page's fitz extraction is treated as
# suspect (font-encoding issues, scanned images) and pdfplumber is
# tried instead. Not a guarantee of more text — some pages (title
# pages, section dividers) are legitimately this short.
_MIN_CHARS_PER_PAGE = 20

# Real English text runs roughly 38-40% vowels among its letters.
# Validated against a real font-encoding-garbled title page (21.3%
# vowels) vs. clean body text from the same class of document (38.5%).
# Below this ratio, treat the page as likely garbled even if it
# returned plenty of characters.
_MIN_VOWEL_RATIO = 0.28

# Skip the vowel-ratio check on pages with too few letters for the
# ratio to mean anything (a short numeric page, a mostly-blank divider).
_MIN_LETTERS_FOR_GARBLE_CHECK = 40

# pdfplumber emits a literal "(cid:N)" token when a glyph can't be
# mapped to a real Unicode character (a broken/missing ToUnicode CMap
# in the PDF's embedded font). Real documents essentially never
# contain this substring, so a handful of occurrences is a precise,
# specific signal — unlike the vowel-ratio check, this catches the
# case even when the repeated "cid" letters coincidentally produce a
# vowel ratio that looks like real English (confirmed on real data:
# "(cid:N)" tokens measured 0.333 — above the 0.28 cutoff above).
_MIN_CID_OCCURRENCES = 3

# A line counts as "bold" only if at least this fraction of its
# characters are bold, not just any single span. Validated on a real
# pattern: a lab-value criterion sentence with just one bolded number
# ("...HBsAg less than 400,000 IU/mL...") measured only 15% bold —
# comfortably below this threshold — while a genuine heading is bold
# essentially end to end.
_MIN_BOLD_FRACTION = 0.5

# A heading candidate must be at least this fraction letters, not
# digits/symbols. Validated on a real survivor of the bold-fraction
# fix: a wholly-bolded lab-value fragment ("<100 x 109/L 100,000/mL")
# measured only 20% letters, versus every real heading tested — even
# ones containing section numbers ("3.1.1 Study Design") — measuring
# 0.69 or higher. Bold-fraction alone can't catch this case, since a
# short fragment that's entirely bold by itself looks identical, by
# that signal, to a genuinely short, wholly-bold heading.
_MIN_LETTER_RATIO = 0.4

# A line is a heading candidate if its font size is at least this much
# larger than the document's body-text baseline. 1.15 is an untested
# starting guess (my synthetic test used a 14pt/10pt = 1.4x heading,
# comfortably above this) — needs validating against real PDFs.
_HEADING_SIZE_RATIO = 1.15

# Safety cap, not the primary signal now that font size/bold does the
# real discriminating — guards against something pathological like an
# entire bold paragraph being treated as one giant "heading".
_MAX_HEADING_WORDS = 12

# A line (heading candidate OR ordinary body text) appearing on more
# than this fraction of the document's pages is running header/footer
# boilerplate. Validated on real data: a document footer repeated on
# ~97% of pages, versus a genuinely recurring real heading ("Study
# Design") that repeated on only ~7% of pages across the same
# document — wide enough margin that 15% safely separates the two
# without risking a real, if infrequent, recurring line.
_MAX_LINE_PAGE_FREQUENCY = 0.15

_PAGE_FOOTER_PATTERN = re.compile(r"(?i)^page \d+ of \d+$")
_SLASH_PAGE_NUMBER_PATTERN = re.compile(r"^\d+/\d+$")  # e.g. "36/84" -- an alternate page-number style
_BARE_SECTION_NUMBER_PATTERN = re.compile(r"^\d+(\.\d+)*\.?$")

# A real content page rarely starts more than a couple of new sections.
# A page contributing more candidates than this is almost certainly a
# table of contents, index, or similar listing page — discard all of
# that page's candidates rather than try to sort real from fake within
# it. Confirmed on real data: a genuine TOC page produced 100+
# candidates in one document; real content pages produce a handful.
_MAX_CANDIDATES_PER_PAGE = 15


def _has_enough_letters(text: str) -> bool:
    """True if at least _MIN_LETTER_RATIO of text's non-space characters are letters."""
    stripped = text.replace(" ", "")
    if not stripped:
        return False
    letters = sum(1 for c in stripped if c.isalpha())
    return (letters / len(stripped)) >= _MIN_LETTER_RATIO


def _has_cid_placeholders(text: str) -> bool:
    """True if `text` contains multiple literal '(cid:N)' glyph-mapping-failure tokens."""
    return text.count("(cid:") >= _MIN_CID_OCCURRENCES


def extract_text(pdf_path: str) -> str:
    """
    Extract all text from a PDF. Thin wrapper around
    extract_text_and_headings() for callers that only need the text —
    same contract and behavior as before this function existed.
    """
    text, _headings = extract_text_and_headings(pdf_path)
    return text


def extract_text_and_headings(pdf_path: str) -> tuple[str, list[str]]:
    """
    Single pass over the PDF that returns both the full document text
    and a list of heading candidates (lines whose font is notably
    larger than the document's body-text baseline, or bold).

    Uses get_text("dict") exclusively — verified to be a strict
    superset of plain get_text() (reconstructing plain text from its
    spans produces identical output), so this replaces what would
    otherwise be two separate read-and-parse passes over the PDF with
    one. The thin-page/garbled-page fallback logic is unchanged from
    before, just applied to the reconstructed text instead of a
    separately-fetched plain string.
    """
    doc = fitz.open(pdf_path)
    try:
        page_lines: list[list[dict]] = []
        all_sizes: list[float] = []
        line_pages: dict[str, set] = defaultdict(set)  # every line's text -> pages it appeared on

        for page_num, page in enumerate(doc):
            data = page.get_text("dict")
            lines_on_page = []
            for block in data["blocks"]:
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    line_text = "".join(s["text"] for s in spans)
                    max_size = max(s["size"] for s in spans)
                    total_len = sum(len(s["text"]) for s in spans)
                    bold_len = sum(
                        len(s["text"]) for s in spans if "bold" in s["font"].lower() or (s["flags"] & 16)
                    )
                    is_bold = bool(total_len) and (bold_len / total_len) >= _MIN_BOLD_FRACTION
                    lines_on_page.append({"text": line_text, "size": max_size, "bold": is_bold})
                    all_sizes.append(round(max_size, 1))
                    stripped = line_text.strip()
                    if stripped:
                        line_pages[stripped].add(page_num)
            page_lines.append(lines_on_page)

        total_pages = len(page_lines)
        body_size = Counter(all_sizes).most_common(1)[0][0] if all_sizes else 0.0

        # Every line document-wide that looks like running boilerplate —
        # not just heading candidates. Reuses _is_boilerplate as-is: a
        # page-footer/bare-number match or an implausible page-frequency
        # disqualifies a line whether it was ever a heading candidate or
        # always just ordinary body text.
        boilerplate_lines = {
            text for text, pages_seen in line_pages.items() if _is_boilerplate(text, pages_seen, total_pages)
        }

        pages_text: list[str] = []
        raw_headings: list[tuple[str, int]] = []  # (text, page_num), before boilerplate filtering
        for page_num, lines_on_page in enumerate(page_lines):
            reconstructed = "\n".join(line["text"] for line in lines_on_page).strip()

            is_thin = len(reconstructed) < _MIN_CHARS_PER_PAGE
            if is_thin or _looks_garbled(reconstructed):
                fallback = _extract_with_pdfplumber(pdf_path, page_num)
                if fallback and (is_thin or not _looks_garbled(fallback)):
                    reconstructed = fallback

            # Boilerplate stripping happens LAST, after the fallback
            # decision, applied to whichever text won -- fitz's or
            # pdfplumber's. Applying it earlier (before choosing
            # fallback) risks stripping a page down to near-nothing,
            # which would itself trigger the is_thin fallback and pull
            # the boilerplate straight back in via pdfplumber, which
            # has no knowledge of boilerplate_lines at all.
            reconstructed = _strip_boilerplate_lines(reconstructed, boilerplate_lines)
            pages_text.append(reconstructed)

            for line in lines_on_page:
                text = line["text"].strip()
                if not text or len(text.split()) > _MAX_HEADING_WORDS:
                    continue
                if not _has_enough_letters(text):
                    continue
                is_large = body_size and line["size"] >= body_size * _HEADING_SIZE_RATIO
                if is_large or line["bold"]:
                    raw_headings.append((text, page_num))
    finally:
        doc.close()

    pages_seen_by_text: dict[str, set] = defaultdict(set)
    for text, page_num in raw_headings:
        pages_seen_by_text[text].add(page_num)

    candidates_per_page = Counter(page_num for _text, page_num in raw_headings)
    toc_pages = {page_num for page_num, count in candidates_per_page.items() if count > _MAX_CANDIDATES_PER_PAGE}

    headings = [
        text
        for text, page_num in raw_headings
        if page_num not in toc_pages and not _is_boilerplate(text, pages_seen_by_text[text], total_pages)
    ]

    return "\n\n".join(pages_text), headings


def _strip_boilerplate_lines(text: str, boilerplate_lines: set) -> str:
    """
    Remove any line from `text` whose stripped content is in
    `boilerplate_lines`. Works on text from either extraction path
    (fitz's reconstruction or pdfplumber's fallback) since both
    return newline-separated text.
    """
    kept = [line for line in text.split("\n") if line.strip() not in boilerplate_lines]
    return "\n".join(kept).strip()


def _is_boilerplate(text: str, pages_seen: set, total_pages: int) -> bool:
    """
    True if `text` is running header/footer noise rather than a real
    heading: a "Page N of M" line, a bare section number with no title
    ("4.2", "8.2.4"), or text appearing on an implausibly large share
    of the document's pages for a one-off section heading.
    """
    if _PAGE_FOOTER_PATTERN.match(text):
        return True
    if _SLASH_PAGE_NUMBER_PATTERN.match(text):
        return True
    if _BARE_SECTION_NUMBER_PATTERN.match(text):
        return True
    if total_pages and len(pages_seen) / total_pages > _MAX_LINE_PAGE_FREQUENCY:
        return True
    return False


def _vowel_ratio(text: str) -> float | None:
    """
    Fraction of vowels among alphabetic characters, or None if there
    aren't enough letters in `text` for the ratio to be meaningful.
    """
    letters = [c for c in text if c.isalpha()]
    if len(letters) < _MIN_LETTERS_FOR_GARBLE_CHECK:
        return None
    vowels = sum(1 for c in letters if c.lower() in "aeiou")
    return vowels / len(letters)


def _looks_garbled(text: str) -> bool:
    """True if `text` shows (cid:N) placeholders, or has a suspiciously low vowel ratio."""
    if _has_cid_placeholders(text):
        return True
    ratio = _vowel_ratio(text)
    return ratio is not None and ratio < _MIN_VOWEL_RATIO


def _extract_with_pdfplumber(pdf_path: str, page_num: int) -> str | None:
    """Re-extract a single page with pdfplumber. Returns None if that's no better."""
    import pdfplumber  # imported lazily — only needed on the fallback path

    with pdfplumber.open(pdf_path) as pdf:
        if page_num >= len(pdf.pages):
            return None
        text = pdf.pages[page_num].extract_text()
        return text.strip() if text else None