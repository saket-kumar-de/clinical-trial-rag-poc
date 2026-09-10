# Clinical Trial RAG POC

A scoped proof-of-concept RAG pipeline over real, public clinical trial
data: ingestion -> structured storage -> chunking/embedding -> retrieval
-> a small serving API.

This is deliberately **not** a production system. It's built to
demonstrate specific techniques (structure-aware chunking, a LangGraph
retrieval node, Aurora + pgvector) at a small, defensible scale, not to
handle real trial volumes or real regulatory requirements.

## Architecture

```
ClinicalTrials.gov / PubMed / protocol PDFs
              |
      extraction & section classification
              |
        +-----+-----+
        |           |
    Aurora        pgvector
  (structured)   (chunked + embedded text)
        |           |
        |      LangGraph retrieval node
        |           |
        +-----+-----+
              |
        FastAPI /search endpoint
```

## Stack

- **Ingestion**: ClinicalTrials.gov API v2, PubMed E-utilities, PyMuPDF/pdfplumber for protocol PDFs
- **Storage**: Aurora PostgreSQL Serverless (Free Tier, Express configuration)
- **Vector search**: pgvector, embeddings via local `sentence-transformers` (no API cost)
- **Retrieval**: LangGraph, single node
- **Serving**: FastAPI

## Setup

1. Create an Aurora PostgreSQL Serverless cluster via Express configuration (AWS Console -> RDS -> Create database -> Aurora PostgreSQL)
2. Set a billing alarm in AWS Budgets before doing anything else
3. Copy `.env.example` to `.env` and fill in your cluster's connection details
4. `pip install -r requirements.txt`
5. Apply the schema: run `src/storage/schema.sql` against your cluster (via `psql` or any Postgres client)
6. Run ingestion: `python -m scripts.run_ingestion --condition "<some condition>"`
7. Start the API: `uvicorn src.api.main:app --reload`

## Status

Scaffold stage — module structure and interfaces are in place; core logic
(marked `TODO` / `NotImplementedError` throughout) is the next step.

## Scope note

Rule-based section classification and a single-node retrieval graph are
deliberate simplifications for this POC's scale. An LLM-based classifier
and a multi-node agentic graph are the natural next steps if this grows
past a handful of trials.
