"""
Connection helper for Aurora PostgreSQL, authenticating via IAM
tokens rather than a static password -- required by the "IAM only"
authentication mode enforced by Aurora's Free Tier Express
configuration. Swap for a connection pool (e.g. psycopg2.pool or
asyncpg) once this moves past POC scale.
"""
import boto3
import psycopg2
from psycopg2.extras import RealDictCursor

from src.config import settings


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
    """
    token = _generate_auth_token()
    return psycopg2.connect(
        host=settings.db_host,
        port=settings.db_port,
        dbname=settings.db_name,
        user=settings.db_user,
        password=token,
        sslmode="require",
        cursor_factory=RealDictCursor,
    )


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