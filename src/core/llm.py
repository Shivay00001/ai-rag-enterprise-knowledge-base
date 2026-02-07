"""
LLM abstraction layer.
Provides unified interface for different LLM providers.
"""
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from src.config import settings


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response for the given prompt."""
        pass
    
    @abstractmethod
    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Stream a response for the given prompt."""
        pass


class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation using GPT models."""
    
    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        self.model = model or settings.llm_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self._client = None
    
    @property
    def client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        return self._client
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response using OpenAI.
        
        Args:
            prompt: The input prompt.
            **kwargs: Additional parameters for the API.
            
        Returns:
            Generated text response.
        """
        messages = [{"role": "user", "content": prompt}]
        
        # Check if system message is provided
        if "system" in kwargs:
            messages.insert(0, {"role": "system", "content": kwargs.pop("system")})
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        
        return response.choices[0].message.content
    
    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """
        Stream a response using OpenAI.
        
        Args:
            prompt: The input prompt.
            **kwargs: Additional parameters.
            
        Yields:
            Chunks of the generated response.
        """
        messages = [{"role": "user", "content": prompt}]
        
        if "system" in kwargs:
            messages.insert(0, {"role": "system", "content": kwargs.pop("system")})
        
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=True,
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class MockLLM(BaseLLM):
    """Mock LLM for testing without API calls."""
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Return a mock response."""
        return f"This is a mock response to: {prompt[:100]}..."
    
    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Stream a mock response."""
        response = await self.generate(prompt, **kwargs)
        for word in response.split():
            yield word + " "


def get_llm(provider: str = "openai") -> BaseLLM:
    """
    Factory function to get the appropriate LLM.
    
    Args:
        provider: The LLM provider to use.
        
    Returns:
        An LLM instance.
    """
    providers = {
        "openai": OpenAILLM,
        "mock": MockLLM,
    }
    
    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(providers.keys())}")
    
    return providers[provider]()
