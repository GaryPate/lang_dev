import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    model_name: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    base_url: str | None = os.getenv("BASE_URL")
    api_key: str | None = os.getenv("API_KEY")
    tavily_api_key: str | None = os.getenv("TAVILY_API_KEY")
    chroma_dir: str = os.getenv("CHROMA_DIR", "./.chroma/healthbot")
    collection_name: str = os.getenv("CHROMA_COLLECTION", "healthbot-medical-kb")
    langfuse_public_key: str | None = os.getenv("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = os.getenv("LANGFUSE_SECRET_KEY")
    langfuse_host: str | None = os.getenv("LANGFUSE_HOST")


settings = Settings()
