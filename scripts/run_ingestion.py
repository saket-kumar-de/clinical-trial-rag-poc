"""
End-to-end ingestion orchestration:
  fetch trials -> extract entities -> chunk -> embed -> write to Aurora

Run once schema.sql has been applied to the Aurora cluster.
Usage:
  python -m scripts.run_ingestion --condition "non-small cell lung cancer"
  python -m scripts.run_ingestion --pdf data/raw/NCT03961204_protocol.pdf --nct-id NCT03961204
"""
import argparse
from pathlib import Path

from src.embedding.chunker import chunk_by_section, chunk_text
from src.embedding.embedder import embed_texts
from src.extraction.entity_extractor import (
    eligibility_from_ctgov_record,
    endpoints_from_ctgov_record,
    interventions_from_ctgov_record,
    trial_from_ctgov_record,
)
from src.ingestion.clinicaltrials_api import fetch_trial_by_id, fetch_trials
from src.ingestion.pdf_downloader import download_protocol_pdf, find_protocol_document
from src.ingestion.pdf_loader import extract_text_and_headings
from src.ingestion.pubmed_api import fetch_abstracts, search_pubmed
from src.storage.db import (
    get_connection,
    insert_chunks,
    insert_eligibility_criteria,
    insert_endpoints,
    insert_interventions,
    insert_trial,
)

# An auto-downloaded PDF producing fewer chunks than this gets flagged
# in the run's summary as worth a manual look, rather than silently
# assuming it extracted correctly -- cheap to compute (just a count),
# doesn't require touching pdf_loader.py's own, already heavily
# tested internals to add a new diagnostic.
_LOW_CHUNK_COUNT_THRESHOLD = 5


def _ingest_trial(record: dict) -> str:
    """
    Extract, chunk, embed, and write one trial's full ClinicalTrials.gov
    record as a single atomic transaction -- commits only if every
    write succeeds, rolls back entirely otherwise, so a failure never
    leaves this trial half-written. Returns the trial's nct_id for the
    caller's progress reporting.

    Only eligibility criteria and endpoint descriptions get chunked
    and embedded. Intervention names are short, structured facts --
    better served by a direct SQL query than semantic search, and
    most wouldn't clear _MIN_CHUNK_CHARS anyway.
    """
    trial = trial_from_ctgov_record(record)
    interventions = interventions_from_ctgov_record(record)
    endpoints = endpoints_from_ctgov_record(record)
    eligibility = eligibility_from_ctgov_record(record)

    text_chunks = []
    for ec in eligibility:
        text_chunks.extend(
            chunk_text(trial.nct_id, "clinicaltrials.gov", "eligibility_criteria", ec.description, min_length=1)
        )
    for ep in endpoints:
        text_chunks.extend(
            chunk_text(trial.nct_id, "clinicaltrials.gov", "endpoints", ep.description, min_length=1)
        )

    embeddings = embed_texts([c.text for c in text_chunks]) if text_chunks else []

    conn = get_connection()
    try:
        insert_trial(trial, conn=conn)
        insert_interventions(interventions, conn=conn)
        insert_endpoints(endpoints, conn=conn)
        insert_eligibility_criteria(eligibility, conn=conn)
        insert_chunks(text_chunks, embeddings, conn=conn)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return trial.nct_id


def _ingest_pubmed_literature(nct_id: str, max_articles: int = 5) -> int:
    """
    Search PubMed for articles referencing this trial's NCT ID, chunk
    and embed their abstracts, and write them as their OWN separate
    atomic transaction -- deliberately not part of _ingest_trial's
    transaction. A PubMed lookup failing, or (very commonly, for newer
    or still-recruiting trials) simply finding nothing, must never
    roll back or block the trial's already-committed structured data.

    Searches both the Secondary Source ID field (where NCT numbers are
    formally indexed) and title/abstract text (since not every paper
    referencing a trial gets that formal indexing done correctly --
    a documented, real gap, not a hypothetical one) rather than an
    unrestricted term search, which would under-count real matches.

    Returns the number of articles ingested (0 if none found).
    """
    query = f"{nct_id}[si] OR {nct_id}[tiab]"
    pubmed_ids = search_pubmed(query, max_results=max_articles)
    if not pubmed_ids:
        return 0

    articles = fetch_abstracts(pubmed_ids)
    if not articles:
        return 0

    text_chunks = []
    for article in articles:
        source_doc = f"pubmed:{article['pubmed_id']}"
        text_chunks.extend(chunk_text(nct_id, source_doc, "literature", article["abstract"], min_length=1))

    embeddings = embed_texts([c.text for c in text_chunks]) if text_chunks else []

    conn = get_connection()
    try:
        insert_chunks(text_chunks, embeddings, conn=conn)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return len(articles)


def _ensure_trial_exists(nct_id: str) -> None:
    """
    Check whether `nct_id` already has a trials row; if not, fetch its
    real metadata from ClinicalTrials.gov and insert it.

    chunks.nct_id is a foreign key -- inserting PDF chunks without a
    real trials row would either fail outright, or (if we inserted a
    placeholder instead) reintroduce the exact stub-data problem this
    project already found and fixed once already: a placeholder title
    silently blocking the real one from ever being written, since
    insert_trial's upsert only fires again if this function is called
    with genuinely new data.

    In practice, the fetch_trial_by_id branch below only ever fires
    via the standalone --pdf/--nct-id CLI path. When called from
    _ingest_pdf_for_trial (the auto-download path), _ingest_trial has
    already inserted this exact trial moments earlier in the same
    loop iteration, so `exists` is always True there and this returns
    immediately -- not dead code, just a branch today's --condition
    test runs never exercised.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM trials WHERE nct_id = %s", (nct_id,))
            exists = cur.fetchone() is not None
    finally:
        conn.close()

    if exists:
        return

    print(f"  No existing trial record for {nct_id} -- fetching from ClinicalTrials.gov...")
    record = fetch_trial_by_id(nct_id)
    trial = trial_from_ctgov_record(record)
    insert_trial(trial)


def _ingest_pdf(pdf_path: str, nct_id: str) -> int:
    """
    Ingest one protocol PDF for a known trial. Structurally different
    from Phases 1/2: there's no API that says "here's the PDF for
    this trial" -- this takes an explicit file path and NCT ID rather
    than being auto-discovered, and only ever touches trials (to
    satisfy the foreign key) and chunks -- interventions/endpoints/
    eligibility_criteria come from the API, not the PDF, by design.

    Returns the number of chunks ingested.
    """
    _ensure_trial_exists(nct_id)

    text, headings = extract_text_and_headings(pdf_path)
    chunks = chunk_by_section(nct_id, Path(pdf_path).name, text, headings)
    embeddings = embed_texts([c.text for c in chunks]) if chunks else []

    insert_chunks(chunks, embeddings)
    return len(chunks)


def _ingest_pdf_for_trial(record: dict, nct_id: str) -> tuple[str, int] | None:
    """
    Find and auto-download this trial's protocol PDF from
    ClinicalTrials.gov's CDN (if it has one), then run it through the
    same PDF pipeline as the manual --pdf path. Returns
    (local_pdf_path, chunk_count) if a protocol document existed, or
    None if this trial simply has no protocol PDF attached -- expected
    to be common, not an error, same as PubMed literature often being
    absent.

    Deliberately not folded into _ingest_trial's transaction, for the
    same reason as _ingest_pubmed_literature: a download or extraction
    failure here must never roll back the trial's already-committed
    structured data.
    """
    doc_info = find_protocol_document(record)
    if doc_info is None:
        return None

    pdf_path = download_protocol_pdf(nct_id, doc_info["filename"])
    chunk_count = _ingest_pdf(pdf_path, nct_id)
    return pdf_path, chunk_count


def main(condition: str, max_results: int, download_pdfs: bool = True, max_pdf_downloads: int = 10) -> None:
    records = fetch_trials(condition, max_results)
    print(f"Fetched {len(records)} trial(s) for condition: {condition!r}")

    succeeded, failed = 0, 0
    pdf_downloads_used = 0
    flagged_for_review = []  # (nct_id, pdf_path, chunk_count) -- low/zero chunk counts, worth a manual look

    for record in records:
        try:
            nct_id = _ingest_trial(record)
            succeeded += 1
            print(f"  [OK]     {nct_id}")
        except Exception as e:
            failed += 1
            # One bad trial shouldn't take down the whole batch --
            # log it and move on to the next record.
            fallback_id = record.get("protocolSection", {}).get("identificationModule", {}).get("nctId", "UNKNOWN")
            print(f"  [FAILED] {fallback_id}: {e}")
            continue

        try:
            article_count = _ingest_pubmed_literature(nct_id)
            if article_count:
                print(f"           + {article_count} PubMed article(s)")
        except Exception as e:
            # Independent of the trial's own success -- a PubMed
            # failure is logged but never flips a trial from
            # succeeded to failed.
            print(f"           PubMed lookup failed for {nct_id}: {e}")

        if download_pdfs and pdf_downloads_used < max_pdf_downloads:
            try:
                result = _ingest_pdf_for_trial(record, nct_id)
                if result:
                    pdf_path, chunk_count = result
                    pdf_downloads_used += 1
                    print(f"           + protocol PDF: {chunk_count} chunk(s) from {Path(pdf_path).name}")
                    if chunk_count < _LOW_CHUNK_COUNT_THRESHOLD:
                        flagged_for_review.append((nct_id, pdf_path, chunk_count))
            except Exception as e:
                # Same independence as the PubMed step -- a download
                # or extraction failure is logged, never rolls back
                # this trial's already-good structured data.
                print(f"           PDF download/ingestion failed for {nct_id}: {e}")

    print(f"\nDone. {succeeded} succeeded, {failed} failed.")
    if download_pdfs:
        print(f"PDF downloads used: {pdf_downloads_used}/{max_pdf_downloads}")
    if flagged_for_review:
        print(f"\n{len(flagged_for_review)} PDF(s) flagged for review (fewer than {_LOW_CHUNK_COUNT_THRESHOLD} chunks):")
        for nct_id, pdf_path, chunk_count in flagged_for_review:
            print(f"  {nct_id}: {chunk_count} chunk(s) -- {pdf_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", help="Ingest ClinicalTrials.gov + PubMed data for this condition")
    parser.add_argument("--max-results", type=int, default=30)
    parser.add_argument("--pdf", help="Path to a protocol PDF to ingest")
    parser.add_argument("--nct-id", help="NCT ID the --pdf belongs to (required together with --pdf)")
    parser.add_argument(
        "--max-pdf-downloads", type=int, default=10, help="Max protocol PDFs to auto-download per run"
    )
    parser.add_argument(
        "--no-pdf-download", action="store_true", help="Disable automatic protocol PDF discovery/download"
    )
    args = parser.parse_args()

    if args.pdf:
        if not args.nct_id:
            parser.error("--nct-id is required when using --pdf")
        count = _ingest_pdf(args.pdf, args.nct_id)
        print(f"Ingested {count} chunk(s) from {args.pdf} for {args.nct_id}")
    elif args.condition:
        main(
            args.condition,
            args.max_results,
            download_pdfs=not args.no_pdf_download,
            max_pdf_downloads=args.max_pdf_downloads,
        )
    else:
        parser.error("provide either --condition (ClinicalTrials.gov + PubMed) or --pdf together with --nct-id")