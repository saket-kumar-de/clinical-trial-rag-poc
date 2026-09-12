"""
Finds and downloads a trial's protocol PDF from ClinicalTrials.gov's
CDN, given the trial's raw API record.

Many trials have no attached protocol document at all -- this is the
common case, not an error condition, same as PubMed literature often
being absent for newer or still-recruiting trials.
"""
from pathlib import Path

import requests

CDN_BASE_URL = "https://cdn.clinicaltrials.gov/large-docs"

# Kept separate from data/raw/ (our two manually-verified, already
# debugged test fixtures) so freshly auto-downloaded, unreviewed files
# never get mixed in with files that have actually been looked at.
DOWNLOAD_DIR = Path("data/downloaded")


def find_protocol_document(record: dict) -> dict | None:
    """
    Return the largeDocs entry for this trial's protocol document
    (hasProtocol=True), or None if the trial has no protocol document
    attached. A trial's document list can include other files (an
    ICF, a SAP alone) -- hasProtocol specifically identifies the
    protocol itself, rather than grabbing the first attached file.
    """
    large_docs = record.get("documentSection", {}).get("largeDocumentModule", {}).get("largeDocs", [])
    for doc in large_docs:
        if doc.get("hasProtocol"):
            return doc
    return None


def download_protocol_pdf(nct_id: str, filename: str) -> str:
    """
    Download a trial's protocol PDF to DOWNLOAD_DIR, returning the
    local file path.

    URL pattern confirmed against real trial documents:
    https://cdn.clinicaltrials.gov/large-docs/<last 2 digits of NCT>/<NCT_ID>/<filename>
    """
    last_two_digits = nct_id[-2:]
    url = f"{CDN_BASE_URL}/{last_two_digits}/{nct_id}/{filename}"

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    local_path = DOWNLOAD_DIR / f"{nct_id}_{filename}"

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    local_path.write_bytes(response.content)

    return str(local_path)