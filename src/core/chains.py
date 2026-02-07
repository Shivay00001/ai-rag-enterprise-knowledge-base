"""
RAG Chain implementation.
Orchestrates retrieval and generation for question answering.
"""
from typing import Any

from src.core.retriever import DocumentRetriever
from src.core.llm import OpenAILLM, get_llm
from src.core.vectorstore import Document
from src.config import settings


# Prompt templates
RAG_SYSTEM_PROMPT = """You are an AI assistant that answers questions based on the provided context.
Your answers should be:
1. Accurate and based ONLY on the provided context
2. Concise but comprehensive
3. If the context doesn't contain enough information, acknowledge this honestly

Context:
{context}
"""

RAG_USER_PROMPT = """Based on the context provided above, please answer the following question:

Question: {question}

Answer:"""


class RAGChain:
    """
    Retrieval-Augmented Generation chain.
    
    Combines document retrieval with LLM generation to answer questions
    based on the knowledge base.
    """
    
    def __init__(
        self,
        retriever: DocumentRetriever,
        llm=None,
        temperature: float | None = None,
    ):
        self.retriever = retriever
        self.llm = llm or get_llm("openai")
        self.temperature = temperature or settings.llm_temperature
    
    def format_context(self, documents: list[Document]) -> str:
        """Format retrieved documents into a context string."""
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            context_parts.append(f"[Document {i}] (Source: {source})\n{doc.page_content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    async def query(self, question: str) -> dict[str, Any]:
        """
        Execute the RAG pipeline.
        
        Args:
            question: The user's question.
            
        Returns:
            Dictionary with answer and source documents.
        """
        # Step 1: Retrieve relevant documents
        documents = await self.retriever.retrieve(question)
        
        # Step 2: Format context
        context = self.format_context(documents)
        
        # Step 3: Construct prompts
        system_prompt = RAG_SYSTEM_PROMPT.format(context=context)
        user_prompt = RAG_USER_PROMPT.format(question=question)
        
        # Step 4: Generate answer
        answer = await self.llm.generate(
            prompt=user_prompt,
            system=system_prompt,
            temperature=self.temperature,
        )
        
        return {
            "answer": answer,
            "source_documents": documents,
            "context_used": context,
        }
    
    async def query_with_history(
        self,
        question: str,
        chat_history: list[dict] | None = None,
    ) -> dict[str, Any]:
        """
        Execute RAG with conversation history for multi-turn conversations.
        
        Args:
            question: The current question.
            chat_history: List of previous messages [{role, content}, ...].
            
        Returns:
            Dictionary with answer and source documents.
        """
        # Condense history into context if present
        history_context = ""
        if chat_history:
            history_str = "\n".join([
                f"{msg['role'].capitalize()}: {msg['content']}"
                for msg in chat_history[-5:]  # Last 5 messages
            ])
            history_context = f"\n\nConversation History:\n{history_str}\n"
        
        # Retrieve documents
        documents = await self.retriever.retrieve(question)
        context = self.format_context(documents)
        
        # Enhanced system prompt with history
        system_prompt = RAG_SYSTEM_PROMPT.format(context=context) + history_context
        user_prompt = RAG_USER_PROMPT.format(question=question)
        
        answer = await self.llm.generate(
            prompt=user_prompt,
            system=system_prompt,
            temperature=self.temperature,
        )
        
        return {
            "answer": answer,
            "source_documents": documents,
        }


class SummarizationChain:
    """Chain for document summarization."""
    
    SUMMARIZE_PROMPT = """Please provide a concise summary of the following document:

{document}

Summary:"""
    
    def __init__(self, llm=None):
        self.llm = llm or get_llm("openai")
    
    async def summarize(self, text: str, max_length: int = 300) -> str:
        """Summarize a document."""
        prompt = self.SUMMARIZE_PROMPT.format(document=text[:10000])  # Limit input
        
        summary = await self.llm.generate(
            prompt=prompt,
            system="You are a helpful assistant that creates clear, concise summaries.",
            max_tokens=max_length,
        )
        
        return summary
