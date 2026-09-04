from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class QueryRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, Any]]] = []
    top_k: Optional[int] = 3
    max_length: Optional[int] = 512

class RawTextUploadRequest(BaseModel):
    title: str
    content: str

class Citation(BaseModel):
    source: str
    page: int
    content: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: List[Citation]

class DocumentInfo(BaseModel):
    filename: str
    pages: int
    chunks: int
    file_size_bytes: int
