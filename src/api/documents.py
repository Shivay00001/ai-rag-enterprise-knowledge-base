"""
Document management API endpoints.
Handles document upload, processing, and management.
"""
import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status, Request
from pydantic import BaseModel

from src.config import settings
from src.api.auth import User, get_current_active_user
from src.ingestion.pipeline import IngestionPipeline
from src.models.documents import DocumentMetadata, DocumentStatus


router = APIRouter()


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: DocumentStatus
    message: str
    chunks_created: int = 0


class DocumentListResponse(BaseModel):
    documents: list[DocumentMetadata]
    total: int
    page: int
    page_size: int


# In-memory document store (replace with database in production)
documents_db: dict[str, DocumentMetadata] = {}


def compute_file_hash(content: bytes) -> str:
    """Compute SHA-256 hash of file content."""
    return hashlib.sha256(content).hexdigest()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    request: Request,
    file: Annotated[UploadFile, File(description="Document to upload")],
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    """
    Upload a document for processing and indexing.
    
    Supported formats: PDF, DOCX, TXT, MD, HTML
    """
    # Validate file extension
    allowed_extensions = {".pdf", ".docx", ".txt", ".md", ".html"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Read file content
    content = await file.read()
    
    # Check file size
    file_size_mb = len(content) / (1024 * 1024)
    if file_size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB"
        )
    
    # Generate document ID
    file_hash = compute_file_hash(content)
    document_id = str(uuid.uuid4())
    
    # Check for duplicate
    for doc in documents_db.values():
        if doc.file_hash == file_hash:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Document already exists in the knowledge base"
            )
    
    # Create metadata
    metadata = DocumentMetadata(
        document_id=document_id,
        filename=file.filename,
        file_type=file_ext,
        file_size_bytes=len(content),
        file_hash=file_hash,
        uploaded_by=current_user.username,
        uploaded_at=datetime.utcnow(),
        status=DocumentStatus.PROCESSING,
    )
    
    documents_db[document_id] = metadata
    
    # Process document
    try:
        vectorstore = request.app.state.vectorstore
        pipeline = IngestionPipeline(vectorstore=vectorstore)
        chunks_created = await pipeline.process_document(
            content=content,
            filename=file.filename,
            document_id=document_id,
            file_type=file_ext,
        )
        
        # Update status
        metadata.status = DocumentStatus.INDEXED
        metadata.chunks_count = chunks_created
        
        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            status=DocumentStatus.INDEXED,
            message="Document processed and indexed successfully",
            chunks_created=chunks_created,
        )
        
    except Exception as e:
        metadata.status = DocumentStatus.FAILED
        metadata.error_message = str(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}"
        )


@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    current_user: Annotated[User, Depends(get_current_active_user)],
    page: int = 1,
    page_size: int = 20,
):
    """List all documents in the knowledge base."""
    all_docs = list(documents_db.values())
    
    # Pagination
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated_docs = all_docs[start_idx:end_idx]
    
    return DocumentListResponse(
        documents=paginated_docs,
        total=len(all_docs),
        page=page,
        page_size=page_size,
    )


@router.get("/{document_id}", response_model=DocumentMetadata)
async def get_document(
    document_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    """Get document metadata by ID."""
    if document_id not in documents_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    return documents_db[document_id]


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    request: Request,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    """Delete a document from the knowledge base."""
    if document_id not in documents_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Delete from vector store
    vectorstore = request.app.state.vectorstore
    await vectorstore.delete_by_document_id(document_id)
    
    # Delete metadata
    del documents_db[document_id]
    
    return {"message": "Document deleted successfully", "document_id": document_id}
