import time
from typing import Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.logging import get_logger
from app.infra.embedding.ollama_embed import get_embeddings
from app.infra.vectorstore.milvus_store import insert_chunks

logger = get_logger("rag_ingest")


def ingest_text(file_id: str, source: str, text: str) -> int:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    pieces = splitter.split_text(text)

    chunks: List[Dict] = []
    now = int(time.time())
    for idx, chunk_text in enumerate(pieces, start=1):
        chunk_uid = f"{file_id}_chunk_{idx:04d}"
        chunks.append(
            {
                "chunk_uid": chunk_uid,
                "file_id": file_id,
                "chunk_index": idx,
                "source": source,
                "created_at": now,
                "text": chunk_text,
            }
        )

    emb = get_embeddings()
    vecs = emb.embed_documents([c["text"] for c in chunks])

    inserted = insert_chunks(chunks=chunks, embeddings=vecs)
    logger.info("ingest ok file_id=%s chunks=%s inserted=%s", file_id, len(chunks), inserted)
    return len(chunks)
