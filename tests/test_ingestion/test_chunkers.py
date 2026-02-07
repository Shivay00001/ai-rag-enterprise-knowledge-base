"""Tests for ingestion components."""
import pytest

from src.ingestion.chunkers import RecursiveCharacterChunker, SentenceChunker
from src.ingestion.loaders import TextLoader, MarkdownLoader


class TestRecursiveCharacterChunker:
    """Tests for the recursive character chunker."""
    
    def test_short_text_single_chunk(self):
        """Short text should result in single chunk."""
        chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=20)
        text = "This is a short text."
        chunks = chunker.split(text)
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_long_text_multiple_chunks(self, sample_text):
        """Long text should be split into multiple chunks."""
        chunker = RecursiveCharacterChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.split(sample_text)
        assert len(chunks) > 1
    
    def test_chunk_size_respected(self, sample_text):
        """Chunks should not exceed chunk_size significantly."""
        chunk_size = 150
        chunker = RecursiveCharacterChunker(chunk_size=chunk_size, chunk_overlap=20)
        chunks = chunker.split(sample_text)
        
        for chunk in chunks:
            # Allow some flexibility for word boundaries
            assert len(chunk) <= chunk_size * 1.5
    
    def test_empty_text(self):
        """Empty text should result in no chunks."""
        chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.split("")
        assert len(chunks) == 0


class TestSentenceChunker:
    """Tests for the sentence chunker."""
    
    def test_split_by_sentences(self):
        """Text should be split at sentence boundaries."""
        chunker = SentenceChunker(chunk_size=100, chunk_overlap=0)
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunker.split(text)
        assert len(chunks) >= 1
    
    def test_respects_chunk_size(self):
        """Chunks should respect the maximum size."""
        chunk_size = 50
        chunker = SentenceChunker(chunk_size=chunk_size, chunk_overlap=0)
        text = "Short. Another short. Third short."
        chunks = chunker.split(text)
        
        for chunk in chunks:
            assert len(chunk) <= chunk_size * 2  # Allow flexibility


class TestLoaders:
    """Tests for document loaders."""
    
    def test_text_loader(self):
        """Test plain text loading."""
        loader = TextLoader()
        content = b"Hello, World!"
        text = loader.load(content)
        assert text == "Hello, World!"
    
    def test_text_loader_utf8(self):
        """Test UTF-8 encoded text."""
        loader = TextLoader()
        content = "Hello, 世界!".encode("utf-8")
        text = loader.load(content)
        assert "世界" in text
    
    def test_markdown_loader(self):
        """Test markdown loading."""
        loader = MarkdownLoader()
        content = b"# Header\n\nThis is **bold** text."
        text = loader.load(content)
        assert "Header" in text
        assert "bold" in text
