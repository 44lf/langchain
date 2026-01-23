from langchain_openai import ChatOpenAI
from app.core.config import settings


def get_deepseek_chat() -> ChatOpenAI:
    if not settings.DEEPSEEK_API_KEY or not settings.DEEPSEEK_BASE_URL:
        raise RuntimeError("DeepSeek 配置缺失：DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL")
    return ChatOpenAI(
        model=settings.DEEPSEEK_MODEL,
        openai_api_key=settings.DEEPSEEK_API_KEY,
        openai_api_base=settings.DEEPSEEK_BASE_URL,
        temperature=0.2,
    )
