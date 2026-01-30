from pathlib import Path
import os

from dotenv import load_dotenv
from minio import Minio

from app.services.minio import MINIOservice
from app.services.rag import RAG


def ensure_bucket(client: Minio, bucket: str) -> None:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)


def main() -> None:
    load_dotenv()

    endpoint = os.getenv("MINIO_ENDPOINT")
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")
    secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
    bucket = os.getenv("MINIO_BUCKET")

    if not all([endpoint, access_key, secret_key, bucket]):
        print("Missing MINIO env vars. Check .env")
        return

    client = Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure,
    )
    ensure_bucket(client, bucket)

    # Prepare a small sample file and upload to MinIO
    local_upload = Path("tmp_rag_upload.txt")
    local_upload.write_text(
        "LangChain is a framework for building LLM applications.\n"
        "RAG stands for Retrieval-Augmented Generation.\n"
        "Milvus is a vector database for similarity search.\n",
        encoding="utf-8",
    )

    object_name = "rag_test.txt"
    MINIOservice().upload_file(str(local_upload), object_name)

    # Build index from the object in MinIO
    download_path = Path("tmp_rag_download.txt")
    chunk_count = RAG.build_index(str(download_path), object_name)
    print(f"Indexed {chunk_count} chunks")

    # Ask a simple question
    answer = RAG.ask("What is RAG?", top_k=3)
    print("Answer:", answer)


if __name__ == "__main__":
    main()
