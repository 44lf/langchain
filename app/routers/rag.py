from fastapi import APIRouter, UploadFile, File
from typing import List
import tempfile
import os

class RAGRouter:
    def __init__(self):
        self.router = APIRouter()

    def get_router(self):
        @self.router.post("/upload")
        async def upload_file(file: UploadFile = File(...)):
            #上传文档，并自动完成RAG流程，上传、切块、向量化、储存
            temp_file_path = f"temp_{file.filename}"
            try:
                with open(temp_file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)

                from app.services.rag import RAG
                chunk_count = RAG.build_index(temp_file_path, file.filename)

                return {
                    "filename": file.filename,
                    "chunk_count": chunk_count,
                    "status": "success",
                    "message": f"File '{file.filename}' uploaded and indexed with {chunk_count} chunks."
                }
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

        @self.router.get("/ask")
        async def rag_ask(question: str, top_k: int = 3):
            #基于RAG的问答接口
            from app.services.rag import RAG
            answer = RAG.ask(question, top_k=top_k)
            return {
                "question": question,
                "answer": answer,
                "top_k": top_k
            }
        
        return self.router
