import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("milvus_store")


# 你可以把 dim 固定为 embedding 维度（bge-m3 常见 1024）
# 为了避免你现在不确定维度：第一次写入时用向量长度动态校验。
_DEFAULT_DIM = 1024


def _connect() -> None:
    connections.connect(
        alias="default",
        host=settings.MILVUS_HOST,
        port=settings.MILVUS_PORT,
    )


def ensure_collection(dim: int = _DEFAULT_DIM) -> Collection:
    _connect()

    # 使用 Milvus “默认db”即可；如果你开启了多db，需要额外 using_database
    name = settings.MILVUS_COLLECTION

    if utility.has_collection(name):
        col = Collection(name)
        return col

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="chunk_uid", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="file_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="created_at", dtype=DataType.INT64),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields=fields, description="RAG chunks collection")
    col = Collection(name=name, schema=schema)

    # 向量索引（先用 IVF_FLAT，稳定好理解；后续你可以换 HNSW）
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 1024},
    }
    col.create_index(field_name="embedding", index_params=index_params)

    # 标量索引（可选，Milvus 对 VARCHAR 的索引支持有限；先不强依赖）
    col.load()
    logger.info("Created collection=%s dim=%s", name, dim)
    return col


def insert_chunks(
    *,
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
) -> int:
    if not chunks:
        return 0
    if len(chunks) != len(embeddings):
        raise ValueError("chunks 与 embeddings 数量不一致")

    dim = len(embeddings[0])
    col = ensure_collection(dim=dim)

    now = int(time.time())

    rows = []
    for c, e in zip(chunks, embeddings):
        rows.append({
            "chunk_uid": c["chunk_uid"],
            "file_id": c["file_id"],
            "chunk_index": int(c["chunk_index"]),
            "source": c.get("source", "unknown"),
            "created_at": int(c.get("created_at", now)),
            "text": c["text"][:8192],
            "embedding": e,
        })

    res = col.insert(rows)

    col.flush()
    col.load()
    return len(res.primary_keys)


def search(
    *,
    query_vec: List[float],
    topk: int,
    file_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    dim = len(query_vec)
    col = ensure_collection(dim=dim)

    expr = None
    if file_id:
        # 注意：Milvus expr 字符串拼接要小心引号
        expr = f'file_id == "{file_id}"'

    search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}

    results = col.search(   
        data=[np.array(query_vec, dtype=np.float32)],
        anns_field="embedding",
        param=search_params,
        limit=topk,
        expr=expr,
        output_fields=["chunk_uid", "file_id", "chunk_index", "source", "text"],
    )

    hits = []
    for hit in results[0]:
        fields = hit.entity.get("chunk_uid"), hit.entity.get("file_id")
        hits.append(
            {
                "score": float(hit.score),
                "chunk_uid": hit.entity.get("chunk_uid"),
                "file_id": hit.entity.get("file_id"),
                "chunk_index": int(hit.entity.get("chunk_index")),
                "source": hit.entity.get("source"),
                "text": hit.entity.get("text"),
            }
        )
    return hits
