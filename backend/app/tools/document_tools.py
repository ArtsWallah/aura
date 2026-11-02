import re
import math
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field

from app.tools.schemas import ToolCategory, ExecutionLocation, PermissionLevel
from app.tools.registry import tool_registry
from app.core.security import validate_safe_path

class ReadDocumentInput(BaseModel):
    file_path: str = Field(description="Path to document file (txt, md, json, py, log).")
    chunk_size: int = Field(default=1000, description="Approximate characters per chunk.")
    chunk_overlap: int = Field(default=150, description="Overlap between consecutive chunks.")

@tool_registry.register(
    name="read_document",
    category=ToolCategory.DOCUMENT,
    description="Reads local documents, extracts plain text, and chunks them into structured sections for downstream reasoning or embedding retrieval.",
    input_model=ReadDocumentInput,
    execution_location=ExecutionLocation.LOCAL,
    permission_level=PermissionLevel.READ_ONLY
)
def read_document(params: ReadDocumentInput) -> dict[str, Any]:
    safe_path = validate_safe_path(params.file_path)
    if not safe_path.exists():
        raise FileNotFoundError(f"File not found: {safe_path.name}")
        
    with open(safe_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    total_chars = len(content)
    total_words = len(content.split())
    
    # Chunking
    chunks = []
    step = params.chunk_size - params.chunk_overlap
    if step <= 0:
        step = params.chunk_size
        
    for i in range(0, total_chars, step):
        chunk_text = content[i:i + params.chunk_size]
        chunks.append({
            "chunk_index": len(chunks),
            "start_char": i,
            "end_char": min(i + params.chunk_size, total_chars),
            "text": chunk_text
        })
        if i + params.chunk_size >= total_chars:
            break

    return {
        "file_name": safe_path.name,
        "total_characters": total_chars,
        "total_words": total_words,
        "chunk_count": len(chunks),
        "chunks": chunks[:20], # limit preview
        "sample_preview": content[:400]
    }


class SearchDocumentChunksInput(BaseModel):
    chunks: list[dict[str, Any]] = Field(description="List of document chunk dictionaries.")
    query: str = Field(description="Search query or semantic question.")
    top_k: int = Field(default=3, description="Number of most relevant chunks to return.")

@tool_registry.register(
    name="search_document_chunks",
    category=ToolCategory.DOCUMENT,
    description="Performs keyword and token-overlap relevance scoring over document chunks to retrieve the most contextually relevant excerpts.",
    input_model=SearchDocumentChunksInput,
    execution_location=ExecutionLocation.LOCAL,
    permission_level=PermissionLevel.READ_ONLY
)
def search_document_chunks(params: SearchDocumentChunksInput) -> dict[str, Any]:
    query_terms = set(re.findall(r"\w+", params.query.lower()))
    if not query_terms:
        return {"query": params.query, "matches": []}
        
    scored_chunks = []
    for chunk in params.chunks:
        text = chunk.get("text", "")
        text_words = re.findall(r"\w+", text.lower())
        if not text_words:
            continue
            
        term_freq = sum(1 for w in text_words if w in query_terms)
        relevance_score = term_freq / (math.sqrt(len(text_words)) + 1e-5)
        
        if term_freq > 0:
            scored_chunks.append({
                "chunk_index": chunk.get("chunk_index"),
                "relevance_score": round(relevance_score, 4),
                "text": text
            })
            
    scored_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
    return {
        "query": params.query,
        "matches": scored_chunks[:params.top_k]
    }
