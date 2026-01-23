from langchain_ollama import OllamaEmbeddings
from app.core.config import settings


def get_embeddings() -> OllamaEmbeddings:
    # 注意：需要本机 Ollama 跑着，并且已 pull 对应模型，例如：ollama pull bge-m3
    return OllamaEmbeddings(
        model=settings.EMBEDDING_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
    )
