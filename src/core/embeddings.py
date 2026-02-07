"""
Embedding model abstraction.
Provides a unified interface for different embedding providers.
"""
from abc import ABC, abstractmethod
from typing import List

from sentence_transformers import SentenceTransformer
import numpy as np

from src.config import settings


class BaseEmbeddings(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass


class SentenceTransformerEmbeddings(BaseEmbeddings):
    """
    Sentence Transformers embeddings using Hugging Face models.
    
    Default model: all-MiniLM-L6-v2 (384 dimensions, fast, good quality)
    """
    
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model
        self._model = None
        self._dimension = None
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            # Get dimension from a test embedding
            test_embedding = self.model.encode(["test"])
            self._dimension = len(test_embedding[0])
        return self._dimension
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of text documents to embed.
            
        Returns:
            List of embeddings, one per document.
        """
        if not texts:
            return []
        
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalization for cosine similarity
        )
        
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query text.
        
        Args:
            text: Query text to embed.
            
        Returns:
            Query embedding.
        """
        embedding = self.model.encode(
            [text],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        
        return embedding[0].tolist()


class OpenAIEmbeddings(BaseEmbeddings):
    """OpenAI embeddings using the text-embedding-3-small model."""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        self._dimension = 1536  # Default for text-embedding-3-small
        self._client = None
    
    @property
    def client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=settings.openai_api_key)
        return self._client
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents using OpenAI API."""
        if not texts:
            return []
        
        response = self.client.embeddings.create(
            input=texts,
            model=self.model_name,
        )
        
        return [item.embedding for item in response.data]
    
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        response = self.client.embeddings.create(
            input=[text],
            model=self.model_name,
        )
        
        return response.data[0].embedding


def get_embeddings(provider: str = "sentence_transformers") -> BaseEmbeddings:
    """
    Factory function to get the appropriate embeddings model.
    
    Args:
        provider: The embedding provider to use.
        
    Returns:
        An embeddings instance.
    """
    providers = {
        "sentence_transformers": SentenceTransformerEmbeddings,
        "openai": OpenAIEmbeddings,
    }
    
    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(providers.keys())}")
    
    return providers[provider]()
