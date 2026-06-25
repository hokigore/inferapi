"""vLLM backend adapter — OpenAI-compatible server."""
from typing import AsyncGenerator, Dict, Any, List, Optional
import httpx
import json
import logging

from .base import Backend, make_sse_chunk, make_completion_sse_chunk

logger = logging.getLogger("inferapi.backends.vllm")


class VLLMBackend(Backend):
    """vLLM backend — uses vLLM's OpenAI-compatible API."""

    def __init__(self, base_url: str = "http://localhost:8001"):
        super().__init__(base_url, "vllm")

    async def list_models(self) -> List[Dict[str, Any]]:
        try:
            resp = await self.client.get("/v1/models")
            resp.raise_for_status()
            data = resp.json()
            return data.get("data", [])
        except Exception as e:
            logger.warning(f"vLLM list_models failed: {e}")
            return []

    async def chat_completion(
        self, model: str, messages: List[Dict], **kwargs
    ) -> Dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            **{k: v for k, v in kwargs.items() if v is not None},
        }
        resp = await self.client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def chat_completion_stream(
        self, model: str, messages: List[Dict], **kwargs
    ) -> AsyncGenerator[str, None]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            **{k: v for k, v in kwargs.items() if v is not None},
        }
        async with self.client.stream(
            "POST", "/v1/chat/completions", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        yield "data: [DONE]\n\n"
                        return
                    try:
                        chunk = json.loads(data)
                        # Pass through vLLM's already-OpenAI-compatible format
                        yield f"data: {json.dumps(chunk)}\n\n"
                    except json.JSONDecodeError:
                        continue

    async def completion(
        self, model: str, prompt: str, **kwargs
    ) -> Dict[str, Any]:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            **{k: v for k, v in kwargs.items() if v is not None},
        }
        resp = await self.client.post("/v1/completions", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def completion_stream(
        self, model: str, prompt: str, **kwargs
    ) -> AsyncGenerator[str, None]:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            **{k: v for k, v in kwargs.items() if v is not None},
        }
        async with self.client.stream(
            "POST", "/v1/completions", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        yield "data: [DONE]\n\n"
                        return
                    try:
                        chunk = json.loads(data)
                        yield f"data: {json.dumps(chunk)}\n\n"
                    except json.JSONDecodeError:
                        continue

    async def embedding(
        self, model: str, input_text: str, **kwargs
    ) -> Dict[str, Any]:
        payload = {
            "model": model,
            "input": input_text,
            **{k: v for k, v in kwargs.items() if v is not None},
        }
        resp = await self.client.post("/v1/embeddings", json=payload)
        resp.raise_for_status()
        return resp.json()
