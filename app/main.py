from fastapi import FastAPI
from app.core.logging import setup_logging, get_logger
from app.domain.models import IngestReq, IngestResp, AskReq, AskResp, Citation
from app.rag.ingest import ingest_text
from app.rag.query import ask

setup_logging()
logger = get_logger("main")

app = FastAPI(title="RAG Lab (DeepSeek + Milvus)")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResp)
def ingest(req: IngestReq):
    chunks = ingest_text(file_id=req.file_id, source=req.source, text=req.text)
    return IngestResp(file_id=req.file_id, chunks=chunks)


@app.post("/ask", response_model=AskResp)
def query(req: AskReq):
    answer, hits = ask(question=req.question, file_id=req.file_id)
    citations = [
        Citation(
            chunk_uid=h["chunk_uid"],
            file_id=h["file_id"],
            chunk_index=h["chunk_index"],
            source=h["source"],
            text_preview=(h["text"][:200] + ("..." if len(h["text"]) > 200 else "")),
        )
        for h in hits
    ]
    return AskResp(answer=answer, citations=citations)
