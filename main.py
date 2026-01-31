from fastapi import FastAPI
from app.routers.llm import LLMRouter
from app.routers.rag import RAGRouter
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()


app.include_router(LLMRouter.router, prefix="/llm_ask", tags=["LLM Ask"])
app.include_router(RAGRouter().get_router(), prefix="/rag", tags=["RAG"])


