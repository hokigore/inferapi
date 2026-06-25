"""Example: Use InferAPI with curl."""
# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "llama3:8b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "llama3:8b",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'

# Text completion
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "llama3:8b",
    "prompt": "The meaning of life is",
    "max_tokens": 50
  }'

# List models
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer your-api-key"

# Embeddings
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "llama3:8b",
    "input": "Hello world"
  }'
