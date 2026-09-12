"""
Connection helper for Aurora PostgreSQL, authenticating via IAM
tokens rather than a static password -- required by the "IAM only"
authentication mode enforced by Aurora's Free Tier Express
configuration. Swap for a connection pool (e.g. psycopg2.pool or
asyncpg) once this moves past POC scale.
"""
import boto3
import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import Json, RealDictCursor

from src.config import settings
from src.embedding.chunker import TextChunk
from src.storage.models import Trial


def _generate_auth_token() -> str:
    """
    Request a short-lived (~15 minute) auth token from AWS, scoped to
    this specific database host, port, and user. Generated fresh on
    every connection -- unlike a password, a token can't be reused
    past its validity window, so this can't be computed once and
    cached the way get_model() caches the embedding model.
    """
    client = boto3.client("rds", region_name=settings.aws_region)
    return client.generate_db_auth_token(
        DBHostname=settings.db_host,
        Port=settings.db_port,
        DBUsername=settings.db_user,
    )


def get_connection():
    """
    Return a new psycopg2 connection to the Aurora cluster, using an
    IAM-generated token in place of a password. SSL is required --
    IAM database authentication is rejected outright over a
    plaintext connection.

    register_vector(conn) teaches this specific connection how to
    serialize a Python list of floats into pgvector's `vector` type --
    without it, inserting or querying a vector column raises a type
    error, since psycopg2 has no built-in idea what a `vector` column
    is.
    """
    token = _generate_auth_token()
    conn = psycopg2.connect(
        host=settings.db_host,
        port=settings.db_port,
        dbname=settings.db_name,
        user=settings.db_user,
        password=token,
        sslmode="require",
        cursor_factory=RealDictCursor,
    )
    register_vector(conn)
    return conn


def run_schema(schema_path: str = "src/storage/schema.sql") -> None:
    """Execute schema.sql against the connected database. Run once at setup."""
    with open(schema_path, "r") as f:
        ddl = f.read()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()
    finally:
        conn.close()


def insert_trial(trial: Trial) -> None:
    """
    Insert one trial row. Must happen before inserting any chunks for
    that trial: chunks.nct_id is a foreign key referencing
    trials.nct_id, so inserting a chunk for a trial that doesn't
    exist yet raises a foreign key violation, not a silent no-op.

    ON CONFLICT DO UPDATE makes this a real upsert: re-running
    ingestion for a trial already in the table refreshes its data,
    rather than silently keeping whatever was inserted first (which
    DO NOTHING would do -- fine for a one-off placeholder, wrong once
    real ingestion needs to overwrite stale or incomplete data).
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO trials (nct_id, title, phase, condition, status, source, raw_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (nct_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    phase = EXCLUDED.phase,
                    condition = EXCLUDED.condition,
                    status = EXCLUDED.status,
                    source = EXCLUDED.source,
                    raw_json = EXCLUDED.raw_json
                """,
                (
                    trial.nct_id,
                    trial.title,
                    trial.phase,
                    trial.condition,
                    trial.status,
                    trial.source,
                    Json(trial.raw_json) if trial.raw_json is not None else None,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def insert_chunks(chunks: list[TextChunk], embeddings: list[list[float]]) -> None:
    """
    Write chunks and their embeddings into the chunks table. `chunks`
    and `embeddings` must be the same length and in the same order --
    the same two lists chunk_by_section()/chunk_text() and
    embed_texts() already produce, paired back together here via zip.

    Requires a matching trials row for each chunk's nct_id to already
    exist (see insert_trial) -- this raises a foreign key violation
    otherwise, rather than silently dropping the chunk.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for chunk, embedding in zip(chunks, embeddings):
                cur.execute(
                    """
                    INSERT INTO chunks (nct_id, source_doc, section_type, chunk_text, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (chunk.nct_id, chunk.source_doc, chunk.section_type, chunk.text, embedding),
                )
        conn.commit()
    finally:
        conn.close()