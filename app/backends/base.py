"""Base backend interface for LLM inference providers."""
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any, Optional, List
import httpx
import json
import time
import uuid


class Backend(ABC):
    """Abstract base class for inference backends."""

    def __init__(self, base_url: str, backend_name: str):
        self.base_url = base_url.rstrip("/")
        self.name = backend_name
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(300.0, connect=10.0),
        )

    @abstractmethod
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models from this backend."""
        pass

    @abstractmethod
    async def chat_completion(
        self, model: str, messages: List[Dict], **kwargs
    ) -> Dict[str, Any]:
        """Non-streaming chat completion."""
        pass

    @abstractmethod
    async def chat_completion_stream(
        self, model: str, messages: List[Dict], **kwargs
    ) -> AsyncGenerator[str, None]:
        """Streaming chat completion — yields SSE chunks."""
        pass

    @abstractmethod
    async def completion(
        self, model: str, prompt: str, **kwargs
    ) -> Dict[str, Any]:
        """Non-streaming text completion."""
        pass

    @abstractmethod
    async def completion_stream(
        self, model: str, prompt: str, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Streaming text completion — yields SSE chunks."""
        pass

    @abstractmethod
    async def embedding(
        self, model: str, input_text: str, **kwargs
    ) -> Dict[str, Any]:
        """Text embedding."""
        pass

    async def close(self):
        await self.client.aclose()


def make_sse_chunk(
    content: str,
    model: str,
    finish_reason: Optional[str] = None,
    role: Optional[str] = None,
) -> str:
    """Build an OpenAI-compatible SSE chunk for streaming."""
    delta = {}
    if role:
        delta["role"] = role
    if content:
        delta["content"] = content

    chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def make_completion_sse_chunk(
    text: str,
    model: str,
    finish_reason: Optional[str] = None,
) -> str:
    """Build an SSE chunk for text completion streaming."""
    chunk = {
        "id": f"cmpl-{uuid.uuid4().hex[:12]}",
        "object": "text_completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": text,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n"
