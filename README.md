# InferAPI — Self-Hosted OpenAI-Compatible Inference Server

A lightweight, drop-in replacement for the OpenAI API that wraps local LLM backends
(vLLM, Ollama) behind a single unified interface. Run your own models, keep your own
data, use any OpenAI-compatible client.

## Features

- **OpenAI-compatible API** — `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings`
- **Multi-backend** — route requests to vLLM, Ollama, or both simultaneously
- **Streaming** — SSE streaming support (`stream: true`)
- **API key auth** — simple bearer token authentication
- **Rate limiting** — per-key request/token throttling
- **Model aliasing** — map `gpt-4` → your local model name
- **Docker-ready** — one command deploy
- **Zero telemetry** — no tracking, no analytics, no phone home

## Quick Start

```bash
# Clone
git clone https://github.com/tioispp/inferapi.git
cd inferapi

# Configure
cp .env.example .env
# Edit .env — set your backends and API keys

# Run with Docker
docker-compose up -d

# Or run directly
pip install -r requirements.txt
python -m app.main
```

## Usage

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-api-key"
)

response = client.chat.completions.create(
    model="gpt-4",  # alias → your local model
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## Configuration

```env
# .env
INFERAPI_HOST=0.0.0.0
INFERAPI_PORT=8000

# API keys (comma-separated)
INFERAPI_API_KEYS=sk-mykey1,sk-mykey2

# Backends
VLLM_BASE_URL=http://localhost:8001
OLLAMA_BASE_URL=http://localhost:11434

# Model aliases (alias:backend:local_name)
INFERAPI_MODEL_ALIASES=gpt-4:vllm:meta-llama/Llama-3-70B,gpt-3.5-turbo:ollama:llama3:8b
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (streaming + non-streaming) |
| `/v1/completions` | POST | Text completions |
| `/v1/models` | GET | List available models |
| `/v1/embeddings` | POST | Text embeddings |
| `/health` | GET | Health check |

## Architecture

```
Client (OpenAI SDK)
    │
    ▼
InferAPI (FastAPI)
    ├── Auth middleware (API key)
    ├── Rate limiter
    ├── Model router (alias → backend)
    │
    ├── vLLM backend
    └── Ollama backend
```

## License

MIT
