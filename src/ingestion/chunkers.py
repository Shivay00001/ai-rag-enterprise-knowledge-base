"""
Text chunking strategies for document processing.
"""
from abc import ABC, abstractmethod
from typing import Iterator
import re

from src.config import settings


class BaseChunker(ABC):
    """Abstract base class for text chunkers."""
    
    @abstractmethod
    def split(self, text: str) -> list[str]:
        """Split text into chunks."""
        pass


class RecursiveCharacterChunker(BaseChunker):
    """
    Recursively splits text using multiple separators.
    
    Tries to split on larger semantic boundaries first (paragraphs),
    then falls back to smaller ones (sentences, words) as needed.
    """
    
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
    
    def split(self, text: str) -> list[str]:
        """
        Split text into chunks with specified size and overlap.
        
        Args:
            text: The text to split.
            
        Returns:
            List of text chunks.
        """
        chunks = []
        self._recursive_split(text, chunks, self.separators)
        return chunks
    
    def _recursive_split(
        self,
        text: str,
        chunks: list[str],
        separators: list[str],
    ):
        """Recursively split text using separators."""
        if len(text) <= self.chunk_size:
            if text.strip():
                chunks.append(text.strip())
            return
        
        # Find the best separator
        separator = separators[0]
        next_separators = separators[1:] if len(separators) > 1 else [""]
        
        if separator:
            parts = text.split(separator)
        else:
            # Character-level split
            parts = list(text)
        
        current_chunk = ""
        
        for part in parts:
            # Add separator back (except for last one)
            candidate = current_chunk + (separator if current_chunk else "") + part
            
            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                # Current chunk is full
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Handle overlap
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + separator + part
                else:
                    current_chunk = part
                
                # If single part is too large, split further
                if len(current_chunk) > self.chunk_size:
                    self._recursive_split(current_chunk, chunks, next_separators)
                    current_chunk = ""
        
        # Add remaining chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())


class SentenceChunker(BaseChunker):
    """
    Splits text by sentences while respecting chunk size limits.
    """
    
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        # Sentence boundary pattern
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    
    def split(self, text: str) -> list[str]:
        """Split text by sentences."""
        sentences = self.sentence_pattern.split(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_length = len(sentence)
            
            if current_length + sentence_length <= self.chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length + 1  # +1 for space
            else:
                # Save current chunk
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and len(current_chunk) > 0:
                    # Keep last few sentences for overlap
                    overlap_sentences = []
                    overlap_length = 0
                    for s in reversed(current_chunk):
                        if overlap_length + len(s) <= self.chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_length += len(s) + 1
                        else:
                            break
                    current_chunk = overlap_sentences + [sentence]
                    current_length = overlap_length + sentence_length
                else:
                    current_chunk = [sentence]
                    current_length = sentence_length
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks


def get_chunker(strategy: str = "recursive") -> BaseChunker:
    """
    Get a chunker by strategy name.
    
    Args:
        strategy: The chunking strategy to use.
        
    Returns:
        A chunker instance.
    """
    chunkers = {
        "recursive": RecursiveCharacterChunker,
        "sentence": SentenceChunker,
    }
    
    if strategy not in chunkers:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(chunkers.keys())}")
    
    return chunkers[strategy]()
