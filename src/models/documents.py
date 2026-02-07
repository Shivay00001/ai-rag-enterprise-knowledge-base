"""
Document models and schemas.
"""
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class DocumentMetadata(BaseModel):
    """Metadata for an indexed document."""
    
    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File extension")
    file_size_bytes: int = Field(..., ge=0, description="File size in bytes")
    file_hash: str = Field(..., description="SHA-256 hash of file content")
    
    uploaded_by: str = Field(..., description="Username of uploader")
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    
    status: DocumentStatus = Field(default=DocumentStatus.PENDING)
    chunks_count: int = Field(default=0, ge=0)
    error_message: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "abc123",
                "filename": "company_policy.pdf",
                "file_type": ".pdf",
                "file_size_bytes": 1048576,
                "file_hash": "sha256...",
                "uploaded_by": "admin",
                "status": "indexed",
                "chunks_count": 42,
            }
        }


class ChunkMetadata(BaseModel):
    """Metadata for a document chunk."""
    
    chunk_id: str
    document_id: str
    chunk_index: int
    total_chunks: int
    source: str
    file_type: str
    ingested_at: datetime
    
    # Optional enrichments
    page_number: Optional[int] = None
    section_title: Optional[str] = None
