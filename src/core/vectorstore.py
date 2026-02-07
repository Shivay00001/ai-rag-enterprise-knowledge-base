"""
Vector store management using ChromaDB.
Handles document storage, retrieval, and similarity search.
"""
import uuid
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import settings
from src.core.embeddings import SentenceTransformerEmbeddings, BaseEmbeddings


class Document:
    """Represents a document chunk with content and metadata."""
    
    def __init__(self, page_content: str, metadata: dict | None = None):
        self.page_content = page_content
        self.metadata = metadata or {}
    
    def __repr__(self):
        return f"Document(content={self.page_content[:50]}..., metadata={self.metadata})"


class VectorStoreManager:
    """
    Manages the ChromaDB vector store for document storage and retrieval.
    
    Features:
    - Persistent storage
    - HNSW index for fast similarity search
    - Metadata filtering
    - Batch operations
    """
    
    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str | None = None,
        embeddings: BaseEmbeddings | None = None,
    ):
        self.persist_directory = persist_directory or settings.chroma_persist_dir
        self.collection_name = collection_name or settings.chroma_collection_name
        self.embeddings = embeddings or SentenceTransformerEmbeddings()
        
        self._client = None
        self._collection = None
    
    async def initialize(self):
        """Initialize the ChromaDB client and collection."""
        # Ensure persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Create ChromaDB client with persistence
        self._client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )
        
        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 200,
                "hnsw:search_ef": 100,
            }
        )
    
    @property
    def collection(self):
        """Get the ChromaDB collection."""
        if self._collection is None:
            raise RuntimeError("VectorStore not initialized. Call initialize() first.")
        return self._collection
    
    async def add_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
    ) -> list[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add.
            ids: Optional list of IDs. Generated if not provided.
            
        Returns:
            List of document IDs.
        """
        if not documents:
            return []
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        
        # Extract texts and metadata
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Generate embeddings
        embeddings = self.embeddings.embed_documents(texts)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        
        return ids
    
    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: dict | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Perform similarity search.
        
        Args:
            query: Query text.
            k: Number of results to return.
            filter_dict: Optional metadata filter.
            
        Returns:
            List of (Document, score) tuples.
        """
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter_dict,
            include=["documents", "metadatas", "distances"],
        )
        
        # Parse results
        documents_with_scores = []
        
        if results["documents"] and results["documents"][0]:
            for i, (text, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )):
                # Convert distance to similarity score (cosine distance to similarity)
                score = 1 - distance
                doc = Document(page_content=text, metadata={**metadata, "score": score})
                documents_with_scores.append((doc, score))
        
        return documents_with_scores
    
    async def delete_by_document_id(self, document_id: str):
        """Delete all chunks belonging to a document."""
        self.collection.delete(
            where={"document_id": document_id}
        )
    
    async def get_collection_stats(self) -> dict:
        """Get statistics about the collection."""
        return {
            "name": self.collection_name,
            "count": self.collection.count(),
            "persist_directory": self.persist_directory,
        }
    
    async def reset(self):
        """Reset the collection (delete all documents)."""
        if self._client:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
