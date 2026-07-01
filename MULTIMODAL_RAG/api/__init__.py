"""FastAPI HTTP surface for ingest, query, and health."""

from config import settings  # noqa: F401 — load .env before app imports

from api.main import app

__all__ = ["app"]