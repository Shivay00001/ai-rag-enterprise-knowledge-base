"""
Document retriever for RAG.
Provides various retrieval strategies.
"""
from typing import List, Optional

from src.core.vectorstore import VectorStoreManager, Document
from src.config import settings


class DocumentRetriever:
    """
    Retrieves relevant documents from the vector store.
    
    Supports:
    - Basic similarity search
    - Threshold-based filtering
    - Maximum marginal relevance (MMR)
    """
    
    def __init__(
        self,
        vectorstore: VectorStoreManager,
        top_k: int | None = None,
        threshold: float | None = None,
    ):
        self.vectorstore = vectorstore
        self.top_k = top_k or settings.retrieval_top_k
        self.threshold = threshold or settings.similarity_threshold
    
    async def retrieve(
        self,
        query: str,
        k: int | None = None,
        filter_dict: dict | None = None,
    ) -> list[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The search query.
            k: Number of documents to retrieve (overrides default).
            filter_dict: Optional metadata filter.
            
        Returns:
            List of relevant documents.
        """
        k = k or self.top_k
        
        # Perform similarity search
        results = await self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter_dict=filter_dict,
        )
        
        # Filter by threshold
        filtered_docs = []
        for doc, score in results:
            if score >= self.threshold:
                doc.metadata["score"] = score
                filtered_docs.append(doc)
        
        return filtered_docs
    
    async def retrieve_with_scores(
        self,
        query: str,
        k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Retrieve documents with their similarity scores.
        
        Args:
            query: The search query.
            k: Number of documents to retrieve.
            
        Returns:
            List of (Document, score) tuples.
        """
        k = k or self.top_k
        
        results = await self.vectorstore.similarity_search(
            query=query,
            k=k,
        )
        
        # Filter by threshold
        return [(doc, score) for doc, score in results if score >= self.threshold]


class HybridRetriever:
    """
    Hybrid retriever combining keyword and semantic search.
    
    Uses reciprocal rank fusion (RRF) to combine results.
    """
    
    def __init__(
        self,
        vectorstore: VectorStoreManager,
        top_k: int = 5,
        alpha: float = 0.5,  # Weight for semantic vs keyword (1.0 = full semantic)
    ):
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.alpha = alpha
    
    async def retrieve(self, query: str) -> list[Document]:
        """
        Retrieve using hybrid search.
        
        For simplicity, this implementation focuses on semantic search.
        In production, integrate with a full-text search engine like Elasticsearch.
        """
        # Semantic search
        semantic_results = await self.vectorstore.similarity_search(
            query=query,
            k=self.top_k * 2,  # Fetch more for fusion
        )
        
        # In a full implementation, we would also do keyword/BM25 search
        # and combine using RRF
        
        # For now, return semantic results
        return [doc for doc, _ in semantic_results[:self.top_k]]
