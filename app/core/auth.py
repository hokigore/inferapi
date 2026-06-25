"""Auth middleware — API key validation."""
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from typing import Optional
import logging

from ..core.config import settings

logger = logging.getLogger("inferapi.auth")

# Paths that don't require auth
PUBLIC_PATHS = {"/health", "/v1/models", "/docs", "/redoc", "/openapi.json"}


def _unauthorized(message: str = "Invalid API key provided.") -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": "invalid_api_key",
            }
        },
    )


class AuthMiddleware(BaseHTTPMiddleware):
    """Validate API key on every request (except public paths)."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Skip auth for public paths
        if path in PUBLIC_PATHS or path.startswith("/docs") or path.startswith("/redoc"):
            return await call_next(request)

        # If no API keys configured → open access (dev mode)
        if not settings.api_keys:
            return await call_next(request)

        # Extract bearer token
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return _unauthorized(
                "Missing or invalid Authorization header. Expected: 'Bearer <api_key>'"
            )

        token = auth_header[7:]

        if token not in settings.api_keys:
            logger.warning(f"Invalid API key attempt: {token[:8]}...")
            return _unauthorized("Invalid API key provided.")

        # Attach key to request state for rate limiting
        request.state.api_key = token

        return await call_next(request)
