"""InferAPI configuration — loaded from env vars."""
import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

logger = logging.getLogger("inferapi.config")


@dataclass
class Settings:
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # Backends
    vllm_base_url: str = "http://localhost:8001"
    ollama_base_url: str = "http://localhost:11434"

    # Rate limiting
    rate_limit: int = 60

    # Parsed from env manually
    api_keys: List[str] = field(default_factory=list)
    model_aliases: Dict[str, Tuple[str, str]] = field(default_factory=dict)


def _load() -> Settings:
    """Load settings from environment + .env file."""
    # Load .env file manually
    env_path = os.path.join(os.getcwd(), ".env")
    if not os.path.exists(env_path):
        env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

    s = Settings(
        host=os.getenv("INFERAPI_HOST", s.host if (s := Settings()) else "0.0.0.0"),
        port=int(os.getenv("INFERAPI_PORT", "8000")),
        log_level=os.getenv("INFERAPI_LOG_LEVEL", "info"),
        vllm_base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8001"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        rate_limit=int(os.getenv("INFERAPI_RATE_LIMIT", "60")),
    )

    # Parse API keys
    keys_str = os.getenv("INFERAPI_API_KEYS", "")
    if keys_str:
        s.api_keys = [k.strip() for k in keys_str.split(",") if k.strip()]

    # Parse model aliases
    aliases_str = os.getenv("INFERAPI_MODEL_ALIASES", "")
    if aliases_str:
        for entry in aliases_str.split(","):
            parts = entry.strip().split(":")
            if len(parts) >= 3:
                alias = parts[0]
                backend = parts[1]
                local_name = ":".join(parts[2:])
                s.model_aliases[alias] = (backend, local_name)

    logger.info(f"Config: {len(s.api_keys)} keys, {len(s.model_aliases)} aliases, "
                f"vllm={s.vllm_base_url}, ollama={s.ollama_base_url}")
    return s


settings = _load()
