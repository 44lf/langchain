from fastapi import APIRouter, UploadFile, File
from typing import List
import tempfile
import os

class RAGRouter:
    def __init__(self):
        self.router = APIRouter()

    def get_router(self):
        from app.services.rag import RAG
# ✅ 修复后的流程
        @self.router.post("/upload")
        async def upload_file(file: UploadFile = File(...)):
            temp_file_path = f"temp_{file.filename}"
            
            try:
                # 步骤1: 保存到本地临时文件
                with open(temp_file_path, "wb") as buffer:
                    buffer.write(await file.read())
                
                # 步骤2: 上传到MinIO ⭐ 新增
                from app.services.minio import MINIOservice
                minio = MINIOservice()
                upload_result = minio.upload_file(temp_file_path, file.filename)
                
                if upload_result.startswith("错误"):
                    return {"status": "failed", "message": upload_result}
                
                # 步骤3: 从MinIO构建索引
                download_path = f"download_{file.filename}"
                chunk_count = RAG.build_index(download_path, file.filename)
                
                # 清理临时下载文件
                if os.path.exists(download_path):
                    os.remove(download_path)
                
                return {
                    "status": "success",
                    "chunk_count": chunk_count
                }
            finally:
                # 清理临时上传文件
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
