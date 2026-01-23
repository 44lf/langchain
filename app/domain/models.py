from typing import List, Optional
from pydantic import BaseModel, Field


class IngestReq(BaseModel):
    file_id: str = Field(..., description="业务文件ID（你自己生成或传uuid）")
    source: str = Field(default="unknown", description="来源：文件名/类型/路径")
    text: str = Field(..., description="要入库的原始文本")


class IngestResp(BaseModel):
    file_id: str
    chunks: int


class AskReq(BaseModel):
    question: str
    file_id: Optional[str] = Field(default=None, description="可选：只在某个file_id内检索")


class Citation(BaseModel):
    chunk_uid: str
    file_id: str
    chunk_index: int
    source: str
    text_preview: str


class AskResp(BaseModel):
    answer: str
    citations: List[Citation]

