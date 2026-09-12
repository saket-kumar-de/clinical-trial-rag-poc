"""
Central configuration loaded from environment variables.
Copy .env.example to .env and fill in real values before running anything.
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # Aurora PostgreSQL connection (IAM authentication -- no static
    # password; see src/storage/db.py for token generation)
    db_host: str = os.getenv("AURORA_HOST", "localhost")
    db_port: int = int(os.getenv("AURORA_PORT", "5432"))
    db_name: str = os.getenv("AURORA_DB", "clinical_rag")
    db_user: str = os.getenv("AURORA_USER", "postgres")
    aws_region: str = os.getenv("AWS_REGION", "ap-south-1")

    # Embedding model
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "384"))

    # PubMed E-utilities (optional API key raises rate limits)
    pubmed_api_key: str = os.getenv("PUBMED_API_KEY", "")
    pubmed_email: str = os.getenv("PUBMED_EMAIL", "")


settings = Settings()