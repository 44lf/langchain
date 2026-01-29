import os

from dotenv import load_dotenv
from langchain_deepseek.chat_models import ChatDeepSeek




class LLMUtils:
    api_key: str = os.getenv("DEEPSEEK_API_KEY")
    base_url: str = os.getenv("DEEPSEEK_base_URL")
    
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is not set in environment variables.")
    if not base_url:
        raise ValueError("DEEPSEEK_base_URL is not set in environment variables.")
    
    @staticmethod
    def init_llm() -> ChatDeepSeek:
        return ChatDeepSeek(
            model="deepseek-chat",
            temperature=0,
            max_retries=3,
            api_key=LLMUtils.api_key,
            base_url=LLMUtils.base_url,
        )


