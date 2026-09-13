# Clinical Trial RAG POC

A working, end-to-end RAG pipeline over real, public clinical trial
data: ingestion -> extraction -> chunking/embedding -> storage ->
retrieval -> a live serving API. Every layer has been tested against
real, live data — not just written and assumed correct.

This is deliberately **not** a production system. It's built to
demonstrate specific techniques (structure-aware chunking, IAM-based
Aurora auth, a branching LangGraph retrieval graph, pgvector +
HNSW) at a small, defensible scale, not to handle real trial volumes
or real regulatory requirements.

**Tech stack**: Python 3.10+ · Aurora PostgreSQL Serverless v2 · pgvector · AWS IAM/`boto3` · `sentence-transformers` · LangGraph · FastAPI · PyMuPDF/pdfplumber

## Architecture

```
ClinicalTrials.gov API      PubMed E-utilities        Protocol PDFs
 (search + single-trial      (search by NCT ID,      (manual --pdf, or
  lookup)                     [si] OR [tiab])          auto-discovered +
        |                          |                    downloaded from
        v                          |                    ClinicalTrials.gov's
 entity_extractor.py               |                    CDN)
 (Trial, Interventions,            |                         |
  Endpoints, Eligibility)          |                    pdf_loader.py
        |                          |                    (extraction, heading
        v                          v                     detection, boilerplate
  chunk_text()               chunk_text()                stripping)
        |                          |                         |
        +-------------+------------+-------------------------+
                       |                                chunk_by_section()
                       v                                      |
                 embed_texts()  <-------------------------------
          (sentence-transformers, all-MiniLM-L6-v2, local, free)
                       |
                       v
         Aurora PostgreSQL + pgvector (IAM authentication)
   trials / interventions / endpoints / eligibility_criteria / chunks
                  (chunks.embedding has an HNSW index)
                       |
                       v
              LangGraph retrieval graph
  classify_query -> retrieve -> deduplicate -> filter_low_confidence
                       -> [confidence branch]
                       |
                       v
              FastAPI /search endpoint
```

### How one `--condition` reaches all three sources

`run_ingestion.py` doesn't take a separate input per source — a single
run fans out from one starting point:

1. `fetch_trials(condition)` gets a batch of matching trials from
   ClinicalTrials.gov
2. **For each trial in that batch**, independently:
   - its structured data (interventions, endpoints, eligibility) is
     extracted and chunked
   - PubMed is searched **using that trial's own NCT ID** as the query
     (`NCT12345678[si] OR NCT12345678[tiab]`), not the original
     condition string
   - its protocol PDF (if one exists) is discovered and downloaded
     from ClinicalTrials.gov's own document CDN, using that same NCT ID

So `--condition "melanoma"` never gets sent to PubMed or the PDF CDN
directly — it only ever selects the initial trial batch. Everything
downstream keys off each individual trial's NCT ID instead. Real
testing (30 melanoma trials) found PubMed literature for about 30% of
trials and a protocol PDF for about 10% — both are commonly absent,
not a sign anything is broken.

## Stack

- **Ingestion**: ClinicalTrials.gov API v2 (batch search + single-trial lookup), PubMed E-utilities, PyMuPDF/pdfplumber for protocol PDFs (with automatic discovery + download from ClinicalTrials.gov's document CDN)
- **Storage**: Aurora PostgreSQL Serverless v2 (Free Tier, Express configuration), **IAM database authentication** (short-lived `boto3`-generated tokens, not a static password — required by Express's Free Tier auth mode)
- **Vector search**: pgvector, HNSW index on `chunks.embedding` (`vector_cosine_ops`), embeddings via local `sentence-transformers` (`all-MiniLM-L6-v2`, no API cost)
- **Retrieval**: LangGraph, a 5-node graph with real conditional branching (see below)
- **Serving**: FastAPI, with declarative request validation (an empty query or an out-of-range `top_k` is rejected automatically)

## Setup

Requires **Python 3.10+** (the codebase uses `str | None` union syntax
throughout, which older versions don't support).

1. Create an Aurora PostgreSQL Serverless cluster via **Express configuration** (AWS Console -> RDS -> Create database -> Aurora PostgreSQL), with max capacity lowered to **4 ACU** to stay within the Free Tier
2. Set a billing alarm in AWS Budgets before doing anything else
3. Express configuration enforces **IAM-only authentication** (no password option) — set this up before continuing:
   - Find your cluster's Resource ID (RDS Console -> your cluster -> Configuration)
   - Create an IAM policy granting `rds-db:connect`, scoped to `arn:aws:rds-db:<region>:<account-id>:dbuser:<resource-id>/postgres`
   - Attach it to an IAM user (a dedicated one, not your root/admin credentials)
   - Run `aws configure` locally with that user's access key
4. Copy `.env.example` to `.env` and fill in your cluster's endpoint, region, and database name (no password — see above)
5. `pip install -r requirements.txt`
6. Apply the schema: `python -c "from src.storage.db import run_schema; run_schema()"`
7. Run ingestion for a condition (ClinicalTrials.gov + PubMed + up to 10 auto-discovered protocol PDFs):
   ```
   python -m scripts.run_ingestion --condition "melanoma" --max-results 30
   ```
   Or ingest a specific protocol PDF manually (download it first — `data/raw/`
   is gitignored and empty on a fresh clone; protocol PDFs follow the pattern
   `https://cdn.clinicaltrials.gov/large-docs/<last-2-digits-of-NCT>/<NCT_ID>/<filename>`):
   ```
   python -m scripts.run_ingestion --pdf data/raw/NCT03961204_protocol.pdf --nct-id NCT03961204
   ```
8. Start the API: `uvicorn src.api.main:app --reload`
9. Try it: open `http://127.0.0.1:8000/docs` (interactive, auto-generated) or:
   ```
   curl "http://127.0.0.1:8000/search?query=What+are+the+eligibility+criteria+for+melanoma&top_k=5"
   ```

## Status

**Core pipeline complete and validated end-to-end against real, live data**, including:

- All three ingestion sources working live, including PubMed's `[si] OR [tiab]` search matching both formally-indexed and text-mentioned trial citations
- PDF extraction validated against sponsors/formats never seen during development, with no corruption or misclassification found
- The full storage layer (all 5 tables), upsert-safe and transaction-shareable, so a multi-step trial ingestion is all-or-nothing rather than partially committed on failure
- A 5-node retrieval graph — query classification (routes to a specific `section_type` when a question clearly asks about one), retrieval (cosine similarity via `pgvector`'s `<=>` operator), near-duplicate removal (by embedding similarity, not a blunt per-trial count cap), per-result confidence filtering, and a conditional low-confidence branch
- A live FastAPI endpoint, tested via `TestClient` and confirmed working against a real running server and real Aurora data

**Not built**: a generation/answer-synthesis step (deliberately out of scope — this is a Data Engineer-focused POC; retrieval quality was prioritized over adding an LLM API dependency that duplicates chat-application territory rather than data engineering).

`scripts/` also contains the diagnostic tools used to validate each layer against real data during development (e.g. `check_embedder.py`, `test_chunking.py`, `retrieval_check.py`) — not part of the pipeline itself, but reusable if any of these components change later.

## Scope notes and known simplifications

- **Confidence and duplicate-similarity thresholds are starting values, not fully calibrated ones.** Real testing already found and fixed one case (a genuine match scoring close to the original cutoff); `eligibility_criteria` and `endpoints` matches also show different natural score ranges, which a single shared threshold doesn't fully account for — a per-`section_type` threshold is the natural next refinement once more real query data accumulates.
- **Section classification is rule-based** (regex/keyword matching), not LLM-based — a deliberate simplification for this scale; a "objective clinical evidence" false-positive match is a known, accepted limitation of this approach.
- **A trial with multiple protocol documents takes only the first one flagged `hasProtocol`** in the API's list order, not necessarily the most recent amendment — a known, low-priority gap (see `pdf_downloader.py`).
- **No generation step** — see Status above.