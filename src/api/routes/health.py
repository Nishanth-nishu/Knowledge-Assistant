"""
Health check endpoints.
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, status
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str
    services: dict[str, Any] | None = None


class ReadinessResponse(BaseModel):
    """Readiness check response."""
    ready: bool
    checks: dict[str, bool]


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Basic health check endpoint for liveness probes.",
)
async def health_check() -> HealthResponse:
    """
    Basic health check.
    
    Returns 200 if the API is running.
    Used for Kubernetes liveness probes.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="0.1.0",
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    status_code=status.HTTP_200_OK,
    summary="Readiness Check",
    description="Detailed readiness check for all dependencies.",
)
async def readiness_check() -> ReadinessResponse:
    """
    Detailed readiness check.
    
    Verifies connectivity to:
    - Qdrant vector database
    - Neo4j graph database
    - Redis cache
    
    Used for Kubernetes readiness probes.
    """
    checks = {
        "qdrant": await _check_qdrant(),
        "neo4j": await _check_neo4j(),
        "redis": await _check_redis(),
    }
    
    all_ready = all(checks.values())
    
    return ReadinessResponse(
        ready=all_ready,
        checks=checks,
    )


async def _check_qdrant() -> bool:
    """Check Qdrant connectivity."""
    try:
        from qdrant_client import QdrantClient
        from src.config import get_qdrant_settings
        
        settings = get_qdrant_settings()
        client = QdrantClient(url=settings.url, timeout=5)
        client.get_collections()
        return True
    except Exception:
        return False


async def _check_neo4j() -> bool:
    """Check Neo4j connectivity."""
    try:
        from neo4j import AsyncGraphDatabase
        from src.config import get_neo4j_settings
        
        settings = get_neo4j_settings()
        driver = AsyncGraphDatabase.driver(
            settings.uri,
            auth=(settings.user, settings.password.get_secret_value()),
        )
        async with driver.session() as session:
            await session.run("RETURN 1")
        await driver.close()
        return True
    except Exception:
        return False


async def _check_redis() -> bool:
    """Check Redis connectivity."""
    try:
        import redis.asyncio as redis
        from src.config import get_redis_settings
        
        settings = get_redis_settings()
        client = redis.from_url(settings.url, socket_timeout=5)
        await client.ping()
        await client.close()
        return True
    except Exception:
        return False
