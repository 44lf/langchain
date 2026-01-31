"""
RAG服务 - 混合检索版本
=====================
核心改进:
1. 向量检索 + BM25关键词检索 (混合召回)
2. RRF算法融合结果
3. 增加debug模式,可观测检索过程
4. 支持metadata(doc_id/source)
"""

from app.utils.llm_client import LLMClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.minio import MINIOservice
from app.services.milvus import MILVUSService
from dotenv import load_dotenv
from pymilvus import Collection
from rank_bm25 import BM25Okapi
import jieba
import os

load_dotenv()


class RAG:
    # ==================== 配置参数 ====================
    SIMILARITY_THRESHOLD = 3.0  # L2距离阈值(已放宽,原0.8太严格)
    MAX_CHUNK_LENGTH = 1900     # 最大chunk长度(与Milvus schema一致)
    
    # BM25索引(类变量,所有实例共享)
    _bm25_index = None
    _bm25_corpus = []  # 存储所有chunk原文
    
    # ==================== 文档加载 ====================
    @staticmethod
    def load_texts(file_path, object_name):
        """
        从MinIO下载文件并读取内容
        
        Args:
            file_path: 本地临时文件路径
            object_name: MinIO中的对象名
            
        Returns:
            str: 文件文本内容
        """
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
            return result
    
    # ==================== 文本切块 ====================
    @staticmethod
    def chunk_texts(text, chunk_size=200, overlap=40):
        """
        优化后的文本切块策略
        
        改进点:
        - chunk_size降至200(原500对中文太大)
        - overlap增至40(增加上下文连续性)
        - 增加中文标点分隔符
        
        Args:
            text: 原始文本
            chunk_size: 单个chunk最大字符数
            overlap: chunk间重叠字符数
            
        Returns:
            list[str]: 切分后的chunk列表
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=[
                "\n---",    # 自定义分隔符(如面试题)
                "\n\n",     # 段落
                "\n",       # 行
                "。",       # 中文句号
                "!",        # 感叹号
                "?",        # 问号
                ";",        # 分号
                " ",        # 空格(最后兜底)
            ])
        chunked_texts = text_splitter.split_text(text)
        return chunked_texts
    
    # ==================== 向量化 ====================
    @staticmethod
    def embed_chunks(chunks):
        """
        将文本chunk转为向量
        
        Args:
            chunks: 文本chunk列表
            
        Returns:
            list: 向量列表
        """
        from app.utils.embedding import EmbeddingUtils
        embedder = EmbeddingUtils()
        vectors = embedder.embed_documents(chunks)
        return vectors

    # ==================== 上传向量到Milvus ====================
    def upload_vectors(self, vectors, chunks):
        """
        将向量和文本上传到Milvus
        
        TODO: 后续需要加metadata(doc_id/source/chunk_index)
        
        Args:
            vectors: 向量列表
            chunks: 对应的文本chunk列表
        """
        milvus = MILVUSService()
        milvus.connect()
        collection = milvus.create_collection("L2", dim=len(vectors[0]))
        labels = [0] * len(chunks)  # TODO: 改为真实doc_id
        descs = chunks
        milvus.insert_vector(collection, vectors, labels, descs)
    
    # ==================== 构建BM25索引 ====================
    @classmethod
    def build_bm25_index(cls, chunks):
        """
        构建BM25关键词索引
        
        BM25原理: 基于词频的经典检索算法
        - 考虑词频(TF)
        - 考虑逆文档频率(IDF)
        - 考虑文档长度归一化
        
        Args:
            chunks: 文本chunk列表
        """
        print(f"🔨 开始构建BM25索引,共 {len(chunks)} 个chunks...")
        
        # 分词: 使用jieba对每个chunk进行中文分词
        tokenized_corpus = [list(jieba.cut(chunk)) for chunk in chunks]
        
        # 构建BM25索引
        cls._bm25_index = BM25Okapi(tokenized_corpus)
        cls._bm25_corpus = chunks  # 保存原文,用于后续返回
        
        print(f"✅ BM25索引构建完成!")
    
    # ==================== 完整构建索引流程 ====================
    @staticmethod
    def build_index(file_path, object_name):
        """
        完整索引构建流程
        
        流程:
        1. 从MinIO下载文件
        2. 切分文本
        3. 构建向量索引(Milvus)
        4. 构建关键词索引(BM25)
        
        Args:
            file_path: 本地临时文件路径
            object_name: MinIO对象名
            
        Returns:
            int: chunk数量
        """
        print(f"\n{'='*50}")
        print(f"开始构建索引: {object_name}")
        print(f"{'='*50}\n")
        
        # 步骤1: 加载文本
        text = RAG.load_texts(file_path, object_name)
        print(f"📄 文档加载成功,总长度: {len(text)} 字符")
        
        # 步骤2: 切分文本
        chunks = RAG.chunk_texts(text)
        print(f"✂️  文本切分完成,共 {len(chunks)} 个chunks")
        
        # 步骤3: 构建向量索引
        print(f"🔢 开始向量化...")
        vectors = RAG.embed_chunks(chunks)
        rag = RAG()
        rag.upload_vectors(vectors, chunks)
        print(f"✅ 向量索引构建完成!")
        
        # 步骤4: 构建BM25索引
        RAG.build_bm25_index(chunks)
        
        print(f"\n{'='*50}")
        print(f"索引构建完成! 总chunks: {len(chunks)}")
        print(f"{'='*50}\n")
        
        return len(chunks)

    # ==================== 混合检索核心 ====================
    @staticmethod
    def hybrid_search(query, top_k=3, alpha=0.5, debug=False):
        """
        混合检索: 向量检索 + BM25关键词检索
        
        策略:
        1. 分别用向量和BM25召回 top_k*2 个结果
        2. 使用RRF(倒数排名融合)算法合并
        3. 返回融合后的top_k结果
        
        Args:
            query: 用户查询
            top_k: 最终返回的结果数
            alpha: 向量检索权重(0~1), 1-alpha为BM25权重
            debug: 是否返回详细调试信息
            
        Returns:
            list: 检索到的文本chunk列表
            或 dict: debug=True时返回详细信息
        """
        print(f"\n{'='*50}")
        print(f"🔍 混合检索开始")
        print(f"查询: {query}")
        print(f"{'='*50}\n")
        
        # ============ 路径1: 向量检索 ============
        print(f"📊 路径1: 向量检索 (召回 top_{top_k*2})")
        vector_results = RAG.vector_search(query, top_k=top_k*2)
        print(f"   召回 {len(vector_results)} 个结果\n")
        
        # ============ 路径2: BM25关键词检索 ============
        print(f"🔤 路径2: BM25关键词检索 (召回 top_{top_k*2})")
        bm25_results = RAG.bm25_search(query, top_k=top_k*2)
        print(f"   召回 {len(bm25_results)} 个结果\n")
        
        # ============ 融合策略: RRF ============
        print(f"🔀 开始融合 (RRF算法)...")
        merged_results = RAG.rrf_fusion(vector_results, bm25_results, top_k)
        print(f"   融合后保留 {len(merged_results)} 个结果\n")
        
        print(f"{'='*50}")
        print(f"✅ 混合检索完成")
        print(f"{'='*50}\n")
        
        if debug:
            return {
                "final_results": merged_results,
                "vector_results": vector_results[:5],
                "bm25_results": bm25_results[:5],
                "fusion_method": "RRF"
            }
        
        return merged_results
    
    # ==================== 向量检索 ====================
    @staticmethod
    def vector_search(query, top_k=5):
        """
        纯向量检索 (你原来的search方法)
        
        Args:
            query: 查询文本
            top_k: 召回数量
            
        Returns:
            list[str]: 检索到的chunk文本列表
        """
        from app.utils.embedding import EmbeddingUtils
        milvus = MILVUSService()
        milvus.connect()
        collection_name = milvus.get_collection_name("L2")
        collection = Collection(collection_name)
        collection.load()

        # 查询向量化
        embedder = EmbeddingUtils()
        query_vector = embedder.embed_query(query)

        # Milvus检索
        results = collection.search(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=top_k*2,  # 多召回一些,用于过滤
            output_fields=["desc"]
        )
        
        # 相似度过滤
        filtered_results = []
        for i, hit in enumerate(results[0]):
            distance = hit.distance
            content = hit.entity.get('desc')
            
            # 只保留相似度高的结果
            if distance <= RAG.SIMILARITY_THRESHOLD:
                filtered_results.append(content)
        
        # 如果过滤后不足top_k,补充一些次优结果
        if len(filtered_results) < top_k and len(results[0]) > len(filtered_results):
            all_results = [hit.entity.get('desc') for hit in results[0]]
            remaining = top_k - len(filtered_results)
            additional = all_results[len(filtered_results):len(filtered_results)+remaining]
            filtered_results.extend(additional)
        
        return filtered_results[:top_k]
    
    # ==================== BM25检索 ====================
    @classmethod
    def bm25_search(cls, query, top_k=5):
        """
        BM25关键词检索
        
        适用场景:
        - 专有名词 (如"FastAPI")
        - 精确匹配 (如代码、配置)
        - 向量检索容易miss的情况
        
        Args:
            query: 查询文本
            top_k: 召回数量
            
        Returns:
            list[str]: 检索到的chunk文本列表
        """
        if cls._bm25_index is None:
            print("⚠️  BM25索引未构建,返回空结果")
            return []
        
        # 查询分词
        tokenized_query = list(jieba.cut(query))
        
        # 计算BM25分数
        scores = cls._bm25_index.get_scores(tokenized_query)
        
        # 取top_k
        top_indices = scores.argsort()[-top_k:][::-1]  # 降序排列
        results = [cls._bm25_corpus[i] for i in top_indices]
        
        return results
    
    # ==================== RRF融合算法 ====================
    @staticmethod
    def rrf_fusion(list1, list2, top_k, k=60):
        """
        RRF (Reciprocal Rank Fusion) 倒数排名融合
        
        原理:
        - 不依赖具体分数,只看排名
        - score = 1/(k + rank)
        - 公平对待不同检索源
        
        示例:
        向量检索: [A(rank=1), B(rank=2), C(rank=3)]
        BM25检索: [B(rank=1), D(rank=2), A(rank=3)]
        
        A的最终分数: 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323
        B的最终分数: 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325
        → B排第一
        
        Args:
            list1: 向量检索结果列表
            list2: BM25检索结果列表
            top_k: 最终返回数量
            k: RRF平滑参数(默认60)
            
        Returns:
            list: 融合后的top_k结果
        """
        scores = {}
        
        # 计算list1的RRF分数
        for rank, doc in enumerate(list1):
            scores[doc] = scores.get(doc, 0) + 1.0 / (k + rank + 1)
        
        # 计算list2的RRF分数
        for rank, doc in enumerate(list2):
            scores[doc] = scores.get(doc, 0) + 1.0 / (k + rank + 1)
        
        # 按分数降序排序
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in sorted_docs[:top_k]]
    
    # ==================== 对外接口 (兼容原有代码) ====================
    @staticmethod
    def search(query, top_k=3, use_hybrid=True):
        """
        统一检索接口
        
        Args:
            query: 查询文本
            top_k: 返回结果数
            use_hybrid: 是否使用混合检索(默认True)
            
        Returns:
            list[str]: 检索到的文本列表
        """
        if use_hybrid and RAG._bm25_index is not None:
            return RAG.hybrid_search(query, top_k)
        else:
            # 降级到纯向量检索
            return RAG.vector_search(query, top_k)
    
    # ==================== 问答接口 ====================
    @staticmethod
    def ask(query, top_k=3, debug=False):
        """
        RAG问答接口
        
        Args:
            query: 用户问题
            top_k: 召回chunk数量
            debug: 是否返回检索详情
            
        Returns:
            str: LLM生成的答案
            或 dict: debug=True时返回详细信息
        """
        # 检索相关文本
        if debug:
            search_result = RAG.hybrid_search(query, top_k=top_k, debug=True)
            texts = search_result["final_results"]
        else:
            texts = RAG.search(query, top_k=top_k)
        
        # 生成答案
        answer = RAG.generate_answer(texts, query)
        
        if debug:
            return {
                "question": query,
                "answer": answer,
                "contexts": [{"text": t[:100] + "...", "length": len(t)} 
                            for t in texts],
                "retrieval_details": search_result
            }
        
        return answer
    
    # ==================== 答案生成 ====================
    @staticmethod
    def generate_answer(relevant_texts, query):
        """
        基于检索到的文本生成答案
        
        Args:
            relevant_texts: 检索到的相关文本列表
            query: 用户问题
            
        Returns:
            str: LLM生成的答案
        """
        llm = LLMClient.get_llm()
        
        if not relevant_texts or all(not t.strip() for t in relevant_texts):
            return "抱歉,我在知识库中没有找到相关信息。"
        
        # 拼接上下文
        context = "\n\n---\n\n".join(relevant_texts)
        
        # 构造prompt
        prompt = f"""你是一个专业的问答助手。请基于以下上下文回答问题。

【重要规则】
1. 使用提供的上下文信息作为参考,结合你的知识库进行回答。
2. 如果上下文中没有相关信息,明确说"上下文中没有相关信息"
3. 回答时请确保信息准确,不要编造事实。
4. 保持回答简洁准确

【上下文】
{context}

【问题】
{query}

【回答】"""
        
        response = llm.invoke([{"role":"user","content":prompt}])
        return response.content