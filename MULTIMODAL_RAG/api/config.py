"""API-layer aliases for central settings (see config/settings.py)."""

from config import settings

BASE_DIR = settings.base_dir
UPLOAD_DIR = settings.upload_dir
CORS_ORIGINS = settings.cors_origins
DEFAULT_TOP_K = settings.rag_top_k
ANSWER_MODEL = settings.openai_answer_model
LANGFUSE_ENABLED = settings.langfuse_enabled