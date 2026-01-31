from app.utils.llm_client import LLMClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.minio import MINIOservice
from app.services.milvus import MILVUSService
from dotenv import load_dotenv
from pymilvus import Collection
load_dotenv()
import os

class RAG:
    @staticmethod
    def load_texts(file_path, object_name):
        """从MinIO下载文件并读取内容"""
        minio = MINIOservice()
        result = minio.download_file(object_name, file_path)
        
        # 检查下载是否成功
        if result.startswith("错误") or result.startswith("S3错误") or result.startswith("未知错误"):
            raise Exception(f"MinIO下载失败: {result}")
        
        # 如果download_file返回的是文件内容(文本),直接返回
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            # 如果返回的是内容字符串
            return result
    
    @staticmethod
    def chunk_texts(text, chunk_size=50, overlap=5):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunked_texts = text_splitter.split_text(text)
        return chunked_texts
    
    @staticmethod
    def embed_chunks(chunks):
        from app.utils.embedding import EmbeddingUtils
        embedder = EmbeddingUtils()
        vectors = embedder.embed_documents(chunks)
        return vectors

    
    def upload_vectors(self, vectors, chunks):
        milvus = MILVUSService()
        milvus.connect()
        collection = milvus.create_collection("L2", dim=len(vectors[0]))
        labels = [0] * len(chunks)
        descs = chunks
        milvus.insert_vector(collection, vectors, labels, descs)
    
    @staticmethod
    def build_index(file_path, object_name):
        """构建索引的完整流程"""
        text = RAG.load_texts(file_path, object_name)
        chunks = RAG.chunk_texts(text)
        vectors = RAG.embed_chunks(chunks)
        rag = RAG()
        rag.upload_vectors(vectors, chunks)
        return len(chunks)

    @staticmethod
    def ask(query, top_k=3):
        texts = RAG.search(query, top_k=top_k)
        return RAG.generate_answer(texts, query)

    @staticmethod
    def search(query, top_k=3):
        from app.utils.embedding import EmbeddingUtils
        milvus = MILVUSService()
        milvus.connect()
        collection_name = milvus.get_collection_name("L2")
        collection = Collection(collection_name)
        collection.load()

        embedder = EmbeddingUtils()
        query_vector = embedder.embed_query(query)

        results = collection.search(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["desc"]
        )

        return [hit.entity.get('desc') for hit in results[0]]
    
    @staticmethod
    def generate_answer(relevant_texts, query):
        llm = LLMClient.get_llm()
        context = "\n".join(relevant_texts)
        prompt = f"Based on the following context:\n{context}\nAnswer the question:\n{query}"
        response = llm.invoke([{"role":"user","content":prompt}])
        return response.content