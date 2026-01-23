import os
from pathlib import Path
from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=_ENV_PATH)


class Settings:
    # DeepSeek (OpenAI-compatible)
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # Milvus
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "127.0.0.1")
    MILVUS_PORT: str = os.getenv("MILVUS_PORT", "19530")
    MILVUS_DB: str = os.getenv("MILVUS_DB", "default")
    MILVUS_COLLECTION: str = os.getenv("MILVUS_COLLECTION", "rag_chunks")

    # Embedding (Ollama default)
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "bge-m3")

    # RAG tuning
    TOPK: int = int(os.getenv("TOPK", "6"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "50"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "10"))


settings = Settings()
