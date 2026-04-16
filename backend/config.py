from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # LLM
    llm_api_key: str = "LLM_API_KEY"
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "openai/gpt-oss-20b"
    llm_temperature: float = 0

    # Embeddings
    embedding_api_key: str = "EMBEDDING_API_KEY"
    embedding_base_url: str = "https://openrouter.ai/api/v1"
    embedding_model: str = "text-embedding-3-small"

    # Resume — set RESUME_PDF_PATH in .env to enable RAG context
    resume_pdf_path: Optional[str] = None

    # CORS
    cors_origins: list[str] = ["*"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
