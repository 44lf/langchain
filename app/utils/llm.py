import os

from dotenv import load_dotenv
from langchain_deepseek.chat_models import ChatDeepseek

load_dotenv()


class LLMUtils:
    @staticmethod
    def init_llm() -> ChatDeepseek:
        return ChatDeepseek(
            model="deepseek-chat",
            temperature=0,
            max_retries=3,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
        )


if __name__ == "__main__":
    llm = LLMUtils.init_llm()
    llm.invoke("Hello, how can I assist you today?")
