"""InferAPI — main application entry point."""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import time
import uuid

from .core.config import settings
from .core.auth import AuthMiddleware
from .backends.registry import registry
from .api.routes import router

# Logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("inferapi")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup + shutdown."""
    logger.info("Starting InferAPI...")
    registry.init()
    logger.info(f"InferAPI running on {settings.host}:{settings.port}")

    yield

    logger.info("Shutting down...")
    await registry.close_all()
    logger.info("Bye.")


app = FastAPI(
    title="InferAPI",
    description="Self-hosted OpenAI-compatible inference server",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(AuthMiddleware)

# Routes
app.include_router(router)


@app.middleware("http")
async def request_id(request: Request, call_next):
    """Add request ID header to all responses."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id

    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{duration:.3f}s"

    logger.info(
        f"{request.method} {request.url.path} "
        f"→ {response.status_code} ({duration:.3f}s)"
    )
    return response


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}


@app.get("/")
async def root():
    """Root — redirect to docs."""
    return {
        "name": "InferAPI",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": ["/v1/chat/completions", "/v1/completions", "/v1/models", "/v1/embeddings"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level,
    )
