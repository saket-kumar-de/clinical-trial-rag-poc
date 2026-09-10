"""Placeholder test — expand as extraction/storage logic gets implemented."""
from src.config import settings


def test_settings_load():
    assert settings.embedding_dim > 0
