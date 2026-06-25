"""Ollama backend adapter — Ollama's native API → OpenAI format."""
from typing import AsyncGenerator, Dict, Any, List, Optional
import httpx
import json
import logging
import uuid
import time

from .base import Backend, make_sse_chunk, make_completion_sse_chunk

logger = logging.getLogger("inferapi.backends.ollama")


class OllamaBackend(Backend):
    """Ollama backend — translates between OpenAI and Ollama formats."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        super().__init__(base_url, "ollama")

    async def list_models(self) -> List[Dict[str, Any]]:
        try:
            resp = await self.client.get("/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = []
            for m in data.get("models", []):
                models.append({
                    "id": m["name"],
                    "object": "model",
                    "created": 0,
                    "owned_by": "ollama",
                })
            return models
        except Exception as e:
            logger.warning(f"Ollama list_models failed: {e}")
            return []

    async def chat_completion(
        self, model: str, messages: List[Dict], **kwargs
    ) -> Dict[str, Any]:
        # Convert OpenAI messages to Ollama format
        ollama_msgs = [{"role": m["role"], "content": m["content"]} for m in messages]

        payload = {
            "model": model,
            "messages": ollama_msgs,
            "stream": False,
            "options": self._build_options(kwargs),
        }

        resp = await self.client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()

        # Convert to OpenAI format
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": data.get("message", {}).get("content", ""),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0)
                + data.get("eval_count", 0),
            },
        }

    async def chat_completion_stream(
        self, model: str, messages: List[Dict], **kwargs
    ) -> AsyncGenerator[str, None]:
        ollama_msgs = [{"role": m["role"], "content": m["content"]} for m in messages]

        payload = {
            "model": model,
            "messages": ollama_msgs,
            "stream": True,
            "options": self._build_options(kwargs),
        }

        # First chunk: send role
        yield make_sse_chunk("", model, role="assistant")

        async with self.client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield make_sse_chunk(content, model)
                    if data.get("done", False):
                        yield make_sse_chunk("", model, finish_reason="stop")
                        yield "data: [DONE]\n\n"
                        return
                except json.JSONDecodeError:
                    continue

    async def completion(
        self, model: str, prompt: str, **kwargs
    ) -> Dict[str, Any]:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": self._build_options(kwargs),
        }

        resp = await self.client.post("/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()

        return {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "text": data.get("response", ""),
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0)
                + data.get("eval_count", 0),
            },
        }

    async def completion_stream(
        self, model: str, prompt: str, **kwargs
    ) -> AsyncGenerator[str, None]:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": self._build_options(kwargs),
        }

        async with self.client.stream("POST", "/api/generate", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    text = data.get("response", "")
                    if text:
                        yield make_completion_sse_chunk(text, model)
                    if data.get("done", False):
                        yield make_completion_sse_chunk("", model, finish_reason="stop")
                        yield "data: [DONE]\n\n"
                        return
                except json.JSONDecodeError:
                    continue

    async def embedding(
        self, model: str, input_text: str, **kwargs
    ) -> Dict[str, Any]:
        payload = {"model": model, "prompt": input_text}
        resp = await self.client.post("/api/embeddings", json=payload)
        resp.raise_for_status()
        data = resp.json()

        return {
            "object": "list",
            "model": model,
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": data.get("embedding", []),
                }
            ],
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }

    def _build_options(self, kwargs: Dict) -> Dict[str, Any]:
        """Map OpenAI params to Ollama options."""
        options = {}
        if "temperature" in kwargs and kwargs["temperature"] is not None:
            options["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs and kwargs["top_p"] is not None:
            options["top_p"] = kwargs["top_p"]
        if "max_tokens" in kwargs and kwargs["max_tokens"] is not None:
            options["num_predict"] = kwargs["max_tokens"]
        if "stop" in kwargs and kwargs["stop"] is not None:
            stop = kwargs["stop"]
            options["stop"] = stop if isinstance(stop, list) else [stop]
        if "seed" in kwargs and kwargs["seed"] is not None:
            options["seed"] = kwargs["seed"]
        if "frequency_penalty" in kwargs and kwargs["frequency_penalty"] is not None:
            options["frequency_penalty"] = kwargs["frequency_penalty"]
        if "presence_penalty" in kwargs and kwargs["presence_penalty"] is not None:
            options["presence_penalty"] = kwargs["presence_penalty"]
        return options
