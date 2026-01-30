from app.utils.llm_client import LLMClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.minio import MINIOservice
from app.services.milvus import MILVUSService
from dotenv import load_dotenv
load_dotenv()
import os

class RAG:
    @staticmethod
    def load_texts(file_path,object_name):
        minio = MINIOservice()
        text = minio.download_file(object_name,file_path)
        return [(text,file_path)]
    
    @staticmethod
    def chunk_texts(text,chunk_size=50, overlap=5):
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
        collection = milvus.create_collection("L2", dim=len(vectors[0]))
        labels = [0] * len(chunks)
        descs = chunks
        milvus.insert_vector(collection, vectors, labels, descs)
    
    @staticmethod
    def search(query,top_k=3):
        from app.utils.embedding import EmbeddingUtils
        milvus = MILVUSService()
        milvus.setup_milvus()
        collection = Collection(os.getenv("MILVUS_DEFAULT_COLLECTION") + "_l2")
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
    
# main():
#     file_path = ""
#     texts = RAG.load_texts(file_path)
#     chunked_texts = RAG.chunk_texts(texts)
#     query = "Your question here"
#     relevant_texts = RAG.search(query)
#     answer = RAG.generate_answer(relevant_texts, query)
#     print("Answer:", answer)