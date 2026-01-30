from fastapi import APIRouter


class RAGRouter:
    def __init__(self):
        self.router = APIRouter()

    def get_router(self):
        async def get_rag():
            return {"message": "RAG endpoint"}