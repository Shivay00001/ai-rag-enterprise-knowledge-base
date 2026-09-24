"""Real OpenAI-compatible chat-completions client. No canned answers.

Reads the API key from the environment. If no key is present, callers get a
clear error instead of fabricated output.
"""
import os

import httpx

BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
API_KEY = os.environ.get("OPENAI_API_KEY", "")


class LLMError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def chat(system: str, user: str, timeout: float = 60.0) -> str:
    """Real HTTP call to an OpenAI-compatible /chat/completions endpoint."""
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise LLMError(
            "No API key configured. Set the OPENAI_API_KEY environment variable "
            "to enable live answers. Refusing to fabricate an answer."
        )
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    try:
        resp = httpx.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.2,
            },
            timeout=timeout,
        )
    except httpx.HTTPError as e:
        raise LLMError(f"LLM request failed at the transport level: {e}")
    if resp.status_code != 200:
        raise LLMError(
            f"LLM provider returned HTTP {resp.status_code}: {resp.text[:500]}",
            status=resp.status_code,
        )
    try:
        return resp.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as e:
        raise LLMError(f"Unexpected LLM response shape: {e}")
