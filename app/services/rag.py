from app.utils.llm_client import LLMClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from minio import MINIOservice
from milvus import MILVUSService
from dotenv import load_dotenv
load_dotenv()
import os

class RAG:
    def load_texts(file_path,object_name):
        minio = MINIOservice()
        text = minio.download_file(file_path,object_name)
        return [(text,file_path)]
    
    def chunk_texts(text,chunk_size=50, overlap=5):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunked_texts = text_splitter.split_text(text)
        return chunked_texts
    
    def embed_chunks(chunks):
        from app.utils.embedding import EmbeddingUtils
        embedder = EmbeddingUtils()
        vectors = embedder.embed_documents(chunks)
        return vectors

    def upload_vectors(vectors,chunks):
        milvus = MILVUSService()
        for i, chunk in enumerate(chunks):
            milvus.insert_vector(os.getenv("MILVUS_DEFAULT_COLLECTION"), chunk)

    def search(query,top_k=3):
        return [relevant_texts]
    
    def generate_answer(relevant_texts, query):
        llm = LLMClient.get_llm()
        context = "\n".join(relevant_texts)
        prompt = f"Based on the following context:\n{context}\nAnswer the question:\n{query}"
        response = llm.invoke([{"role":"user","content":prompt}])
        return response.content
    
# main():
#     file_path = ""
#     texts = RAG.load_texts(file_path)
#     chunked_texts = RAG.chunk_texts(texts)
#     query = "Your question here"
#     relevant_texts = RAG.search(query)
#     answer = RAG.generate_answer(relevant_texts, query)
#     print("Answer:", answer)