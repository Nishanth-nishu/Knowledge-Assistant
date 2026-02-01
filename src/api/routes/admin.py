"""
Admin endpoints for system management.
"""

from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

router = APIRouter()


# =============================================================================
# Models
# =============================================================================

class SystemStatus(BaseModel):
    """System status response."""
    status: str
    version: str
    uptime_seconds: float
    components: dict[str, Any]


class IndexStats(BaseModel):
    """Index statistics."""
    collection_name: str
    document_count: int
    chunk_count: int
    vector_dimension: int
    index_size_mb: float


class ModelInfo(BaseModel):
    """Model information."""
    name: str
    type: str  # "llm", "embedding", "reranker"
    provider: str
    status: str


class ReindexRequest(BaseModel):
    """Reindex request."""
    document_ids: list[str] | None = None
    force: bool = False


# =============================================================================
# Endpoints
# =============================================================================

@router.get(
    "/status",
    response_model=SystemStatus,
    summary="System Status",
    description="Get detailed system status and component health.",
)
async def get_system_status() -> SystemStatus:
    """
    Get comprehensive system status.
    """
    import time
    
    # TODO: Track actual uptime
    start_time = getattr(get_system_status, '_start_time', None)
    if start_time is None:
        get_system_status._start_time = time.time()
        start_time = get_system_status._start_time
    
    uptime = time.time() - start_time
    
    return SystemStatus(
        status="operational",
        version="0.1.0",
        uptime_seconds=uptime,
        components={
            "api": "healthy",
            "qdrant": "healthy",
            "neo4j": "healthy",
            "redis": "healthy",
            "llm": "healthy",
        },
    )


@router.get(
    "/index/stats",
    response_model=IndexStats,
    summary="Index Statistics",
    description="Get vector index statistics.",
)
async def get_index_stats() -> IndexStats:
    """
    Get statistics about the vector index.
    """
    # TODO: Implement actual stats from Qdrant
    return IndexStats(
        collection_name="enterprise_docs",
        document_count=0,
        chunk_count=0,
        vector_dimension=1024,
        index_size_mb=0.0,
    )


@router.get(
    "/models",
    response_model=list[ModelInfo],
    summary="List Models",
    description="List all configured models.",
)
async def list_models() -> list[ModelInfo]:
    """
    List configured models.
    """
    from src.config import get_settings
    
    settings = get_settings()
    
    return [
        ModelInfo(
            name=settings.llm.model,
            type="llm",
            provider=settings.llm.provider,
            status="active",
        ),
        ModelInfo(
            name=settings.embedding.model,
            type="embedding",
            provider="sentence-transformers",
            status="active",
        ),
        ModelInfo(
            name=settings.reranker.model,
            type="reranker",
            provider="colbert",
            status="active",
        ),
    ]


@router.post(
    "/index/reindex",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger Reindexing",
    description="Trigger reindexing of documents.",
)
async def trigger_reindex(request: ReindexRequest) -> dict[str, str]:
    """
    Trigger document reindexing.
    
    Can reindex specific documents or all documents.
    """
    # TODO: Implement reindexing job
    return {
        "status": "accepted",
        "message": "Reindexing job queued",
        "job_id": "placeholder",
    }


@router.post(
    "/cache/clear",
    status_code=status.HTTP_200_OK,
    summary="Clear Cache",
    description="Clear the query cache.",
)
async def clear_cache() -> dict[str, str]:
    """
    Clear all cached queries and responses.
    """
    # TODO: Implement cache clearing
    return {
        "status": "success",
        "message": "Cache cleared",
    }


@router.get(
    "/config",
    summary="Get Configuration",
    description="Get current system configuration (non-sensitive).",
)
async def get_config() -> dict[str, Any]:
    """
    Get non-sensitive configuration values.
    """
    from src.config import get_settings
    
    settings = get_settings()
    
    return {
        "retrieval": {
            "bm25_weight": settings.retrieval.bm25_weight,
            "dense_weight": settings.retrieval.dense_weight,
            "sparse_weight": settings.retrieval.sparse_weight,
            "rrf_k": settings.retrieval.rrf_k,
            "default_top_k": settings.retrieval.default_top_k,
        },
        "agent": {
            "max_iterations": settings.agent.max_iterations,
            "timeout_seconds": settings.agent.timeout_seconds,
            "enable_self_validation": settings.agent.enable_self_validation,
            "enable_multi_hop": settings.agent.enable_multi_hop,
        },
        "document": {
            "chunk_size": settings.document.chunk_size,
            "chunk_overlap": settings.document.chunk_overlap,
            "chunk_strategy": settings.document.chunk_strategy,
            "max_upload_size_mb": settings.document.max_upload_size_mb,
        },
        "features": {
            "knowledge_graph": settings.features.knowledge_graph,
            "multi_agent": settings.features.multi_agent,
            "streaming": settings.features.streaming,
            "caching": settings.features.caching,
        },
    }
