from app.utils.llm_client import LLMClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.minio import MINIOservice
from app.services.milvus import MILVUSService
from dotenv import load_dotenv
from pymilvus import Collection
load_dotenv()
import os

class RAG:
    SIMILARITY_THRESHOLD = 0.8  # L2距离阈值,根据实际调整
    
    # ✅ 新增：最大chunk长度(与Milvus schema一致)
    MAX_CHUNK_LENGTH = 1900  # 留100字符余量
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
    def chunk_texts(text, chunk_size=500, overlap=50):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=[
                        "\n---",      # 面试题的分隔符
                        "\n\n",       # 段落
                        "\n",         # 行
                        "。",         # 中文句号
                        " ",          # 空格
                        ])
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
            limit=top_k*2,
            output_fields=["desc"]
        )
        filtered_results = []
        print(f"\n===== 检索调试信息 =====")
        print(f"查询: {query}")
        for i, hit in enumerate(results[0]):
            distance = hit.distance
            content = hit.entity.get('desc')
            print(f"\n结果 {i+1}:")
            print(f"  距离: {distance:.4f}")
            print(f"  内容前100字: {content[:100]}...")
            
            # 过滤相似度低的结果
            if distance <= RAG.SIMILARITY_THRESHOLD:
                filtered_results.append(content)
            else:
                print(f"  [已过滤: 相似度过低]")
        
        print(f"\n过滤后结果数: {len(filtered_results)}/{len(results[0])}")
        print("=" * 50 + "\n")

        if len(filtered_results) < top_k and len(results[0]) > len(filtered_results):
            # 添加一些未达到阈值但相对较近的结果
            all_results = [hit.entity.get('desc') for hit in results[0]]
            remaining_count = top_k - len(filtered_results)
            additional_results = [all_results[i] for i in range(len(filtered_results), min(len(all_results), len(filtered_results) + remaining_count))]
            filtered_results.extend(additional_results)
        
        return filtered_results[:top_k] if filtered_results else [hit.entity.get('desc') for hit in results[0][:top_k]]
    
    @staticmethod
    def generate_answer(relevant_texts, query):
        llm = LLMClient.get_llm()
        if not relevant_texts or all(not t.strip() for t in relevant_texts):
            return "抱歉,我在知识库中没有找到相关信息。"
        context = "\n\n---\n\n".join(relevant_texts)
        prompt = f"""你是一个专业的问答助手。请基于以下上下文回答问题。

                    【重要规则】
                    1. 使用提供的上下文信息作为参考，结合你的知识库进行回答。
                    2. 如果上下文中没有相关信息,明确说"上下文中没有相关信息"
                    3. 回答时请确保信息准确，不要编造事实。
                    4. 保持回答简洁准确

                    【上下文】
                    {context}

                    【问题】
                    {query}

                    【回答】"""
        response = llm.invoke([{"role":"user","content":prompt}])
        return response.content