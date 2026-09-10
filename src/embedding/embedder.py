"""
Embeds text chunks using a local sentence-transformers model
(no API key, no per-call cost).
"""
from sentence_transformers import SentenceTransformer

from src.config import settings

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Lazily load the embedding model once per process."""
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model_name)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts, returning one vector per input string.
    TODO: batch size tuning if this ever needs to scale past POC volume.
    """
    model = get_model()
    return model.encode(texts, normalize_embeddings=True).tolist()
