from dotenv import load_dotenv
# from langchain.agents import create_agent
from langchain_deepseek.chat_models import ChatDeepseek 
import os

load_dotenv()

class LLMUtils:
    # @staticmethod
    # def create_agent(llm):
    #     agent = create_agent(
    #         model=llm,
    #         system="你是一个乐于助人的助手")
    #     return agent


    def init_llm():
        llm = ChatDeepseek(
            model="deepseek-chat",
            temperature=0,
            max_retries=3,
            api_key=os.getenv("DEEPSEEK_API_KEY")
        )
        return llm
    


llm = init_llm()
ll.invoke('Hello, how can I assist you today?')
