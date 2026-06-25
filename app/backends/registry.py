"""Backend registry — manages backend instances and model routing."""
from typing import Dict, Optional, List, Any
import logging

from .base import Backend
from .vllm import VLLMBackend
from .ollama import OllamaBackend
from ..core.config import settings

logger = logging.getLogger("inferapi.backends")


class BackendRegistry:
    """Routes model requests to the correct backend."""

    def __init__(self):
        self.backends: Dict[str, Backend] = {}
        self.aliases: Dict[str, tuple] = {}

    def init(self):
        """Initialize backends from config."""
        # Always create both — they're cheap, lazy connection
        self.backends["vllm"] = VLLMBackend(settings.vllm_base_url)
        self.backends["ollama"] = OllamaBackend(settings.ollama_base_url)
        self.aliases = settings.model_aliases.copy()

        logger.info(
            f"Backends initialized: vllm={settings.vllm_base_url}, "
            f"ollama={settings.ollama_base_url}"
        )
        if self.aliases:
            logger.info(f"Model aliases: {list(self.aliases.keys())}")

    def resolve_model(self, model: str) -> tuple[str, str]:
        """Resolve model name to (backend_name, local_model_name).

        If model is an alias → return mapped backend + local name.
        If model has no alias → try to detect backend from model name pattern.
        """
        # Check aliases first
        if model in self.aliases:
            backend_name, local_name = self.aliases[model]
            if backend_name in self.backends:
                return backend_name, local_name

        # Auto-detect: if model contains "/" → likely vLLM (HF model ID)
        if "/" in model:
            return "vllm", model

        # Otherwise → ollama (short names like "llama3:8b")
        return "ollama", model

    def get_backend(self, name: str) -> Optional[Backend]:
        return self.backends.get(name)

    async def list_all_models(self) -> List[Dict[str, Any]]:
        """List models from all backends + aliases."""
        models = []

        # Add aliases as models
        for alias in self.aliases:
            models.append({
                "id": alias,
                "object": "model",
                "created": 0,
                "owned_by": "inferapi",
            })

        # Add models from each backend
        for name, backend in self.backends.items():
            try:
                backend_models = await backend.list_models()
                models.extend(backend_models)
            except Exception as e:
                logger.warning(f"Failed to list models from {name}: {e}")

        return models

    async def close_all(self):
        for backend in self.backends.values():
            await backend.close()


# Singleton
registry = BackendRegistry()
