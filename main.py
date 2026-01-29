from fastapi import FastAPI
from app.routers.llm import LLMRouter
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()


app.include_router(LLMRouter.router, prefix="/llm_ask", tags=["LLM Ask"])




