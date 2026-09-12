-- Aurora PostgreSQL schema for the clinical trial RAG POC
-- Run after connecting to your Aurora cluster (via psql or any Postgres client)

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS trials (
    nct_id          TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    phase           TEXT,
    condition       TEXT,
    status          TEXT,
    source          TEXT,          -- 'clinicaltrials.gov', 'ctis', etc.
    raw_json        JSONB,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS interventions (
    id                  SERIAL PRIMARY KEY,
    nct_id              TEXT REFERENCES trials(nct_id),
    name                TEXT NOT NULL,
    intervention_type   TEXT,          -- drug, device, behavioral, etc.
    UNIQUE (nct_id, name)
);

CREATE TABLE IF NOT EXISTS endpoints (
    id              SERIAL PRIMARY KEY,
    nct_id          TEXT REFERENCES trials(nct_id),
    endpoint_type   TEXT,           -- primary, secondary
    description     TEXT NOT NULL,
    UNIQUE (nct_id, description)
);

CREATE TABLE IF NOT EXISTS eligibility_criteria (
    id              SERIAL PRIMARY KEY,
    nct_id          TEXT REFERENCES trials(nct_id),
    criterion_type  TEXT,           -- inclusion, exclusion
    description     TEXT NOT NULL,
    UNIQUE (nct_id, description)
);

-- Chunked + embedded text for RAG retrieval
CREATE TABLE IF NOT EXISTS chunks (
    id              SERIAL PRIMARY KEY,
    nct_id          TEXT REFERENCES trials(nct_id),
    source_doc      TEXT,           -- filename or URL of the source document
    section_type    TEXT,           -- objectives, eligibility_criteria, endpoints, etc.
    chunk_text      TEXT NOT NULL,
    embedding       vector(384),    -- matches EMBEDDING_DIM in .env
    created_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE (nct_id, source_doc, chunk_text)
);

-- HNSW index for fast approximate nearest-neighbor search
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);