"""API routes — OpenAI-compatible endpoints."""
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Dict, Any, List
import json
import logging
import httpx

from ..core.schemas import (
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
    ChatCompletionResponse,
    CompletionResponse,
    EmbeddingResponse,
    ModelListResponse,
    ModelInfo,
    ErrorResponse,
    ErrorDetail,
)
from ..backends.registry import registry

logger = logging.getLogger("inferapi.api")
router = APIRouter(prefix="/v1")


def error_response(message: str, error_type: str = "invalid_request_error",
                   code: str = None, status: int = 400):
    return JSONResponse(
        status_code=status,
        content=ErrorResponse(
            error=ErrorDetail(message=message, type=error_type, code=code)
        ).model_dump(),
    )


@router.get("/models")
async def list_models():
    """List all available models across backends."""
    models = await registry.list_all_models()
    return ModelListResponse(data=[ModelInfo(**m) for m in models])


@router.post("/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    """Create a chat completion — OpenAI-compatible."""
    # Resolve model to backend
    backend_name, local_model = registry.resolve_model(body.model)
    backend = registry.get_backend(backend_name)

    if not backend:
        return error_response(
            f"Backend '{backend_name}' not available",
            code="backend_unavailable",
            status=503,
        )

    # Convert messages to dict format
    messages = [
        {"role": m.role.value, "content": m.content, "name": m.name}
        for m in body.messages
    ]
    # Clean None values
    messages = [{k: v for k, v in m.items() if v is not None} for m in messages]

    # Build kwargs
    kwargs = {
        "temperature": body.temperature,
        "top_p": body.top_p,
        "max_tokens": body.max_tokens,
        "stop": body.stop,
        "seed": body.seed,
        "presence_penalty": body.presence_penalty,
        "frequency_penalty": body.frequency_penalty,
    }

    try:
        if body.stream:
            return StreamingResponse(
                backend.chat_completion_stream(local_model, messages, **kwargs),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            result = await backend.chat_completion(local_model, messages, **kwargs)
            return JSONResponse(content=result)

    except httpx.ConnectError:
        return error_response(
            f"Backend {backend_name} is not reachable. Is it running?",
            error_type="api_error",
            code="backend_connection_error",
            status=503,
        )
    except httpx.HTTPStatusError as e:
        return error_response(
            f"Backend error: {e.response.text}",
            error_type="api_error",
            status=e.response.status_code,
        )
    except Exception as e:
        logger.error(f"Chat completion error: {e}", exc_info=True)
        return error_response(
            f"Internal error: {str(e)}",
            error_type="internal_error",
            status=500,
        )


@router.post("/completions")
async def completions(request: Request, body: CompletionRequest):
    """Create a text completion — OpenAI-compatible."""
    backend_name, local_model = registry.resolve_model(body.model)
    backend = registry.get_backend(backend_name)

    if not backend:
        return error_response(
            f"Backend '{backend_name}' not available",
            code="backend_unavailable",
            status=503,
        )

    prompt = body.prompt if isinstance(body.prompt, str) else body.prompt[0]

    kwargs = {
        "temperature": body.temperature,
        "top_p": body.top_p,
        "max_tokens": body.max_tokens,
        "stop": body.stop,
        "seed": body.seed,
        "echo": body.echo,
        "presence_penalty": body.presence_penalty,
        "frequency_penalty": body.frequency_penalty,
    }

    try:
        if body.stream:
            return StreamingResponse(
                backend.completion_stream(local_model, prompt, **kwargs),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            result = await backend.completion(local_model, prompt, **kwargs)
            return JSONResponse(content=result)

    except httpx.ConnectError:
        return error_response(
            f"Backend {backend_name} is not reachable.",
            error_type="api_error",
            code="backend_connection_error",
            status=503,
        )
    except httpx.HTTPStatusError as e:
        return error_response(
            f"Backend error: {e.response.text}",
            error_type="api_error",
            status=e.response.status_code,
        )
    except Exception as e:
        logger.error(f"Completion error: {e}", exc_info=True)
        return error_response(f"Internal error: {str(e)}", error_type="internal_error", status=500)


@router.post("/embeddings")
async def embeddings(request: Request, body: EmbeddingRequest):
    """Create embeddings — OpenAI-compatible."""
    backend_name, local_model = registry.resolve_model(body.model)
    backend = registry.get_backend(backend_name)

    if not backend:
        return error_response(
            f"Backend '{backend_name}' not available",
            code="backend_unavailable",
            status=503,
        )

    input_text = body.input if isinstance(body.input, str) else body.input[0]

    try:
        result = await backend.embedding(local_model, input_text)
        return JSONResponse(content=result)
    except httpx.ConnectError:
        return error_response(
            f"Backend {backend_name} is not reachable.",
            error_type="api_error",
            code="backend_connection_error",
            status=503,
        )
    except Exception as e:
        logger.error(f"Embedding error: {e}", exc_info=True)
        return error_response(f"Internal error: {str(e)}", error_type="internal_error", status=500)
