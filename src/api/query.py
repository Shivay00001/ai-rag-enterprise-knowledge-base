"""
RAG Query API endpoints.
Handles semantic search and question answering over the knowledge base.
"""
from typing import Annotated
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from src.config import settings
from src.api.auth import User, get_current_active_user
from src.core.chains import RAGChain
from src.core.retriever import DocumentRetriever


router = APIRouter()


class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    question: str = Field(..., min_length=3, max_length=1000, description="The question to ask")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")
    include_sources: bool = Field(default=True, description="Include source documents in response")
    temperature: float = Field(default=0.1, ge=0.0, le=1.0, description="LLM temperature")


class SourceDocument(BaseModel):
    """A source document used to generate the answer."""
    document_id: str
    chunk_id: str
    content: str
    score: float
    metadata: dict = {}


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    answer: str
    sources: list[SourceDocument] = []
    query: str
    processing_time_ms: float
    model_used: str


class SearchRequest(BaseModel):
    """Request model for semantic search."""
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=10, ge=1, le=50)
    threshold: float = Field(default=0.0, ge=0.0, le=1.0)


class SearchResponse(BaseModel):
    """Response model for semantic search."""
    results: list[SourceDocument]
    total: int
    query: str


@router.post("/", response_model=QueryResponse)
async def query_knowledge_base(
    request: Request,
    query_request: QueryRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    """
    Query the knowledge base using RAG.
    
    This endpoint:
    1. Retrieves relevant document chunks using semantic search
    2. Constructs a prompt with the retrieved context
    3. Generates an answer using the configured LLM
    """
    start_time = datetime.utcnow()
    
    try:
        vectorstore = request.app.state.vectorstore
        
        # Initialize retriever and RAG chain
        retriever = DocumentRetriever(
            vectorstore=vectorstore,
            top_k=query_request.top_k,
            threshold=settings.similarity_threshold,
        )
        
        rag_chain = RAGChain(
            retriever=retriever,
            temperature=query_request.temperature,
        )
        
        # Execute query
        result = await rag_chain.query(query_request.question)
        
        # Calculate processing time
        end_time = datetime.utcnow()
        processing_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Format sources
        sources = []
        if query_request.include_sources:
            for doc in result.get("source_documents", []):
                sources.append(SourceDocument(
                    document_id=doc.metadata.get("document_id", "unknown"),
                    chunk_id=doc.metadata.get("chunk_id", "unknown"),
                    content=doc.page_content[:500],  # Truncate for response
                    score=doc.metadata.get("score", 0.0),
                    metadata=doc.metadata,
                ))
        
        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            query=query_request.question,
            processing_time_ms=processing_time_ms,
            model_used=settings.llm_model,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )


@router.post("/search", response_model=SearchResponse)
async def semantic_search(
    request: Request,
    search_request: SearchRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    """
    Perform semantic search without LLM generation.
    
    Returns the most similar documents to the query.
    """
    try:
        vectorstore = request.app.state.vectorstore
        
        # Perform similarity search
        results = await vectorstore.similarity_search(
            query=search_request.query,
            k=search_request.top_k,
        )
        
        # Filter by threshold and format
        sources = []
        for doc, score in results:
            if score >= search_request.threshold:
                sources.append(SourceDocument(
                    document_id=doc.metadata.get("document_id", "unknown"),
                    chunk_id=doc.metadata.get("chunk_id", "unknown"),
                    content=doc.page_content,
                    score=score,
                    metadata=doc.metadata,
                ))
        
        return SearchResponse(
            results=sources,
            total=len(sources),
            query=search_request.query,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/stats")
async def get_query_stats(
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    """Get query statistics and system information."""
    return {
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "retrieval_top_k": settings.retrieval_top_k,
        "similarity_threshold": settings.similarity_threshold,
    }
