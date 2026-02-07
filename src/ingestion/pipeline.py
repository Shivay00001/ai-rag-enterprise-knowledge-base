"""
Document ingestion pipeline.
Orchestrates loading, chunking, and indexing of documents.
"""
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from src.config import settings
from src.ingestion.loaders import get_loader
from src.ingestion.chunkers import get_chunker
from src.core.vectorstore import VectorStoreManager, Document


logger = structlog.get_logger(__name__)


class IngestionPipeline:
    """
    End-to-end document ingestion pipeline.
    
    Pipeline stages:
    1. Load: Extract text from document
    2. Chunk: Split into manageable pieces
    3. Enrich: Add metadata
    4. Index: Store in vector database
    """
    
    def __init__(
        self,
        vectorstore: VectorStoreManager,
        chunk_strategy: str = "recursive",
    ):
        self.vectorstore = vectorstore
        self.chunker = get_chunker(chunk_strategy)
    
    async def process_document(
        self,
        content: bytes,
        filename: str,
        document_id: str,
        file_type: str,
        metadata: dict | None = None,
    ) -> int:
        """
        Process a document through the full ingestion pipeline.
        
        Args:
            content: Raw document bytes.
            filename: Original filename.
            document_id: Unique document identifier.
            file_type: File extension (e.g., ".pdf").
            metadata: Optional additional metadata.
            
        Returns:
            Number of chunks created.
        """
        start_time = datetime.utcnow()
        
        logger.info(
            "Starting document ingestion",
            document_id=document_id,
            filename=filename,
            file_type=file_type,
            size_bytes=len(content),
        )
        
        # Stage 1: Load document
        try:
            loader = get_loader(file_type)
            text = loader.load(content)
            logger.info(
                "Document loaded",
                document_id=document_id,
                text_length=len(text),
            )
        except Exception as e:
            logger.error(
                "Document loading failed",
                document_id=document_id,
                error=str(e),
            )
            raise
        
        # Stage 2: Chunk document
        chunks = self.chunker.split(text)
        logger.info(
            "Document chunked",
            document_id=document_id,
            num_chunks=len(chunks),
        )
        
        # Stage 3: Create Document objects with metadata
        documents = []
        base_metadata = metadata or {}
        
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{document_id}_chunk_{i}"
            
            doc_metadata = {
                **base_metadata,
                "document_id": document_id,
                "chunk_id": chunk_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "source": filename,
                "file_type": file_type,
                "ingested_at": datetime.utcnow().isoformat(),
            }
            
            documents.append(Document(
                page_content=chunk_text,
                metadata=doc_metadata,
            ))
        
        # Stage 4: Index in vector store
        chunk_ids = [doc.metadata["chunk_id"] for doc in documents]
        await self.vectorstore.add_documents(documents, ids=chunk_ids)
        
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()
        
        logger.info(
            "Document ingestion complete",
            document_id=document_id,
            chunks_created=len(chunks),
            processing_time_seconds=processing_time,
        )
        
        return len(chunks)
    
    async def process_batch(
        self,
        file_paths: list[Path],
        parallel: bool = False,
    ) -> dict[str, Any]:
        """
        Process multiple documents.
        
        Args:
            file_paths: List of file paths to process.
            parallel: Whether to process in parallel.
            
        Returns:
            Summary of processed documents.
        """
        results = {
            "total": len(file_paths),
            "successful": 0,
            "failed": 0,
            "documents": [],
        }
        
        for file_path in file_paths:
            document_id = str(uuid.uuid4())
            
            try:
                content = file_path.read_bytes()
                chunks = await self.process_document(
                    content=content,
                    filename=file_path.name,
                    document_id=document_id,
                    file_type=file_path.suffix.lower(),
                )
                
                results["successful"] += 1
                results["documents"].append({
                    "document_id": document_id,
                    "filename": file_path.name,
                    "chunks": chunks,
                    "status": "success",
                })
                
            except Exception as e:
                results["failed"] += 1
                results["documents"].append({
                    "document_id": document_id,
                    "filename": file_path.name,
                    "status": "failed",
                    "error": str(e),
                })
                logger.error(
                    "Batch processing failed for document",
                    filename=file_path.name,
                    error=str(e),
                )
        
        return results
