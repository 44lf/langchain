from app.utils.llm_client import LLMClient

class RAG:
    def load_texts(file_path):
        return [(text,source)]
    
    def chunk_texts(text,chunk_size=50, overlap=5):
        return [chunked_texts]
    
    def search(query,top_k=3):
        return [relevant_texts]
    
    def generate_answer(relevant_texts, query):
        llm = LLMClient.get_llm()
        context = "\n".join(relevant_texts)
        prompt = f"Based on the following context:\n{context}\nAnswer the question:\n{query}"
        response = llm.invoke([{"role":"user","content":prompt}])
        return response.content