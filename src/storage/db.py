"""
Thin connection helper for Aurora PostgreSQL.
Swap for a connection pool (e.g. psycopg2.pool or asyncpg) once this
moves past POC scale.
"""
import psycopg2
from psycopg2.extras import RealDictCursor

from src.config import settings


def get_connection():
    """Return a new psycopg2 connection to the Aurora cluster."""
    return psycopg2.connect(settings.db_url, cursor_factory=RealDictCursor)


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
