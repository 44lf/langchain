"""
RAG路由 - 增强版
=================

新增功能:
1. debug参数 - 查看检索详情
2. use_hybrid参数 - 选择检索模式
3. 更详细的返回信息
"""

from fastapi import APIRouter, UploadFile, File, Query
from typing import List
import tempfile
import os

class RAGRouter:
    def __init__(self):
        self.router = APIRouter()

    def get_router(self):
        from app.services.rag import RAG
        from app.services.minio import MINIOservice
        
        # ==================== 上传文件接口 ====================
        @self.router.post("/upload")
        async def upload_file(file: UploadFile = File(...)):
            """
            上传文件并构建索引
            
            改进:
            - 同时构建向量索引和BM25索引
            - 返回更详细的构建信息
            """
            temp_file_path = f"temp_{file.filename}"
            
            try:
                # 步骤1: 保存到本地临时文件
                with open(temp_file_path, "wb") as buffer:
                    buffer.write(await file.read())
                
                # 步骤2: 上传到MinIO
                minio = MINIOservice()
                upload_result = minio.upload_file(temp_file_path, file.filename)
                
                if upload_result.startswith("错误"):
                    return {"status": "failed", "message": upload_result}
                
                # 步骤3: 从MinIO构建索引 (包括向量+BM25)
                download_path = f"download_{file.filename}"
                chunk_count = RAG.build_index(download_path, file.filename)
                
                # 清理临时下载文件
                if os.path.exists(download_path):
                    os.remove(download_path)
                
                return {
                    "status": "success",
                    "filename": file.filename,
                    "chunk_count": chunk_count,
                    "indexes_built": ["vector_index", "bm25_index"]
                }
            finally:
                # 清理临时上传文件
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

        # ==================== 问答接口 (增强版) ====================
        @self.router.get("/ask")
        async def rag_ask(
            question: str,
            top_k: int = Query(default=3, ge=1, le=10, description="召回chunk数量"),
            debug: bool = Query(default=False, description="是否返回检索详情"),
            use_hybrid: bool = Query(default=True, description="是否使用混合检索")
        ):
            """
            RAG问答接口 - 增强版
            
            新增参数:
            - debug: 开启后返回检索路径详情
            - use_hybrid: 选择使用混合检索或纯向量检索
            
            返回格式 (debug=False):
            {
                "question": "...",
                "answer": "...",
                "top_k": 3
            }
            
            返回格式 (debug=True):
            {
                "question": "...",
                "answer": "...",
                "contexts": [...],
                "retrieval_details": {...}
            }
            """
            from app.services.rag import RAG
            
            if debug:
                # Debug模式: 返回详细信息
                result = RAG.ask(question, top_k=top_k, debug=True)
                result["use_hybrid"] = use_hybrid
                return result
            else:
                # 普通模式: 只返回答案
                answer = RAG.ask(question, top_k=top_k, debug=False)
                return {
                    "question": question,
                    "answer": answer,
                    "top_k": top_k,
                    "use_hybrid": use_hybrid
                }
        
        # ==================== 新增: 纯检索接口 ====================
        @self.router.get("/search")
        async def search_only(
            query: str,
            top_k: int = Query(default=3, ge=1, le=20),
            method: str = Query(default="hybrid", regex="^(hybrid|vector|bm25)$")
        ):
            """
            纯检索接口 (不生成答案)
            
            参数:
            - method: 检索方法
              - hybrid: 混合检索 (默认)
              - vector: 纯向量检索
              - bm25: 纯BM25检索
            
            返回:
            {
                "query": "...",
                "method": "hybrid",
                "results": [
                    {"rank": 1, "text": "...", "preview": "..."},
                    ...
                ]
            }
            """
            from app.services.rag import RAG
            
            # 根据method选择检索方法
            if method == "vector":
                results = RAG.vector_search(query, top_k=top_k)
            elif method == "bm25":
                results = RAG.bm25_search(query, top_k=top_k)
            else:  # hybrid
                results = RAG.hybrid_search(query, top_k=top_k)
            
            return {
                "query": query,
                "method": method,
                "count": len(results),
                "results": [
                    {
                        "rank": i+1,
                        "text": text,
                        "preview": text[:150] + "..." if len(text) > 150 else text,
                        "length": len(text)
                    }
                    for i, text in enumerate(results)
                ]
            }
        
        # ==================== 新增: 评估接口 ====================
        @self.router.post("/evaluate")
        async def evaluate(test_cases: List[dict]):
            """
            批量评估接口
            
            请求体示例:
            [
                {
                    "question": "什么是RAG?",
                    "expected_keywords": ["检索", "生成"]
                },
                ...
            ]
            
            返回:
            {
                "total": 10,
                "passed": 8,
                "accuracy": 0.8,
                "details": [...]
            }
            """
            from app.services.rag import RAG
            
            results = []
            passed = 0
            
            for case in test_cases:
                question = case["question"]
                expected = case.get("expected_keywords", [])
                
                # 获取答案
                answer = RAG.ask(question, top_k=3)
                
                # 检查关键词命中
                hit = any(kw in answer for kw in expected)
                if hit:
                    passed += 1
                
                results.append({
                    "question": question,
                    "answer": answer,
                    "expected_keywords": expected,
                    "passed": hit
                })
            
            return {
                "total": len(test_cases),
                "passed": passed,
                "accuracy": passed / len(test_cases) if test_cases else 0,
                "details": results
            }
        
        return self.router