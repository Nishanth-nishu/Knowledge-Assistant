"""
Enterprise Knowledge Assistant - FastAPI Application.

Production-grade API for intelligent document querying with Agentic RAG.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, ORJSONResponse
from prometheus_client import make_asgi_app

from src.config import get_settings

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown of:
    - Database connections (Qdrant, Neo4j, Redis)
    - LLM clients
    - Background workers
    """
    settings = get_settings()
    
    # Startup
    logger.info("Starting Enterprise Knowledge Assistant API")
    
    # Initialize connections
    try:
        # These will be lazy-loaded when first accessed
        logger.info("API startup complete", 
                   version="0.1.0",
                   environment="development")
    except Exception as e:
        logger.error("Startup failed", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down API")
    
    # Close LLM clients
    from src.models.llm_client import LLMClientFactory
    await LLMClientFactory.close_all()
    
    logger.info("API shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="Enterprise Knowledge Assistant",
        description="""
        Production-grade intelligent assistant for enterprise document analysis.
        
        ## Features
        - 🔍 Three-way hybrid retrieval (BM25 + Dense + SPLADE)
        - 🤖 Multi-agent agentic RAG
        - 📊 Knowledge graph integration
        - 🔐 Enterprise security (JWT, RBAC, rate limiting)
        
        ## Authentication
        Use JWT Bearer tokens or API keys for authentication.
        """,
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=settings.api.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register routes
    _register_routes(app)
    
    # Register exception handlers
    _register_exception_handlers(app)
    
    # Mount Prometheus metrics endpoint
    if settings.monitoring.enable_metrics:
        metrics_app = make_asgi_app()
        app.mount("/metrics", metrics_app)
    
    return app


def _register_routes(app: FastAPI) -> None:
    """Register API routes."""
    from src.api.routes import query, documents, admin, health
    
    # Health check routes (no auth required)
    app.include_router(
        health.router,
        prefix="",
        tags=["Health"],
    )
    
    # API v1 routes
    app.include_router(
        query.router,
        prefix="/api/v1",
        tags=["Query"],
    )
    
    app.include_router(
        documents.router,
        prefix="/api/v1",
        tags=["Documents"],
    )
    
    app.include_router(
        admin.router,
        prefix="/api/v1/admin",
        tags=["Admin"],
    )


def _register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers."""
    
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        logger.warning("Validation error", error=str(exc), path=request.url.path)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc), "type": "validation_error"},
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled exception", 
                    error=str(exc), 
                    path=request.url.path,
                    exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error", "type": "internal_error"},
        )


# Create app instance
app = create_app()


def run_server() -> None:
    """Run the API server (entry point for CLI)."""
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        workers=settings.api.workers,
        reload=settings.api.reload,
        log_level=settings.api.log_level,
    )


if __name__ == "__main__":
    run_server()
