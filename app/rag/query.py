from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.core.config import settings
from app.core.logging import get_logger
from app.infra.embedding.ollama_embed import get_embeddings
from app.infra.llm.deepseek import get_deepseek_chat
from app.infra.vectorstore.milvus_store import search

logger = get_logger("rag_query")


_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个严谨的企业级RAG助手。仅基于给定的【上下文】回答。"
            # "如果上下文不足以回答，直接说“根据现有资料无法确定”。"
            "回答要简洁、可执行，不要编造。",
        ),
        (
            "human",
            "【上下文】\n{context}\n\n【问题】\n{question}\n\n"
            "请给出答案，并尽量在答案中引用上下文要点。",
        ),
    ]
)


def _format_context(hits: List[dict]) -> str:
    blocks = []
    for h in hits:
        blocks.append(
            f"[{h['chunk_uid']}] (file_id={h['file_id']}, idx={h['chunk_index']}, score={h['score']:.4f})\n"
            f"{h['text']}"
        )
    return "\n\n---\n\n".join(blocks)


def ask(question: str, file_id: Optional[str] = None):
    emb = get_embeddings()
    qvec = emb.embed_query(question)

    hits = search(query_vec=qvec, topk=settings.TOPK, file_id=file_id)
    context = _format_context(hits)

    chain = _RAG_PROMPT | get_deepseek_chat() | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    return answer, hits
