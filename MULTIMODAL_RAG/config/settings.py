"""Load .env and expose application settings."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _env(key: str, default: str | None = None) -> str | None:
    value = os.getenv(key)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_bool(key: str, default: bool = False) -> bool:
    raw = _env(key)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_int(key: str, default: int) -> int:
    raw = _env(key)
    if raw is None:
        return default
    return int(raw)


class Settings:
    """Central configuration (populated from environment / .env)."""

    # Paths
    base_dir: Path = BASE_DIR
    upload_dir: Path = Path(
        _env("UPLOAD_DIR", str(BASE_DIR / "data" / "uploads")) or ""
    )
    image_output_dir: Path | None = (
        Path(p) if (p := _env("IMAGE_OUTPUT_DIR")) else None
    )

    # PostgreSQL
    database_url: str | None = _env("DATABASE_URL")

    # OpenAI
    openai_api_key: str | None = _env("OPENAI_API_KEY")
    openai_vision_model: str = _env("OPENAI_VISION_MODEL", "gpt-4o-mini") or "gpt-4o-mini"
    openai_router_model: str = _env("OPENAI_ROUTER_MODEL", "gpt-4o-mini") or "gpt-4o-mini"
    openai_answer_model: str = _env("OPENAI_ANSWER_MODEL", "gpt-4o-mini") or "gpt-4o-mini"
    openai_table_description_model: str = (
        _env("OPENAI_TABLE_DESCRIPTION_MODEL", "gpt-4o-mini") or "gpt-4o-mini"
    )
    openai_embedding_model: str = (
        _env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        or "text-embedding-3-small"
    )
    embedding_dimensions: int = _env_int("EMBEDDING_DIMENSIONS", 1536)

    # Parser
    unstructured_strategy: str = _env("UNSTRUCTURED_STRATEGY", "fast") or "fast"
    use_llm_table_descriptions: bool = _env_bool("USE_LLM_TABLE_DESCRIPTIONS", True)

    # RAG / API
    rag_top_k: int = _env_int("RAG_TOP_K", 5)
    cors_origins: list[str] = [
        o.strip()
        for o in (
            _env(
                "CORS_ORIGINS",
                "http://localhost:3000,http://127.0.0.1:3000",
            )
            or ""
        ).split(",")
        if o.strip()
    ]
    api_host: str = _env("API_HOST", "0.0.0.0") or "0.0.0.0"
    api_port: int = _env_int("API_PORT", 8000)
    api_reload: bool = _env_bool("API_RELOAD", True)

    # Langfuse
    langfuse_public_key: str | None = _env("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = _env("LANGFUSE_SECRET_KEY")
    langfuse_host: str = (
        _env("LANGFUSE_HOST", "https://cloud.langfuse.com")
        or "https://cloud.langfuse.com"
    )

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


settings = Settings()