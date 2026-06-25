"""Example: Use InferAPI with the OpenAI Python SDK."""
from openai import OpenAI

# Point to your InferAPI instance
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-api-key",  # or any string if INFERAPI_API_KEYS not set
)

# Chat completion
response = client.chat.completions.create(
    model="llama3:8b",  # Ollama model
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ],
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="llama3:8b",
    messages=[{"role": "user", "content": "Tell me a joke."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()

# Text completion
response = client.completions.create(
    model="llama3:8b",
    prompt="The capital of France is",
    max_tokens=10,
)
print(response.choices[0].text)

# Embeddings
response = client.embeddings.create(
    model="llama3:8b",
    input="Hello world",
)
print(f"Embedding dim: {len(response.data[0].embedding)}")
