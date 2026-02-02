"""
Observability Package.

Provides:
- Prometheus metrics
- OpenTelemetry tracing
- Logging configuration
"""

from src.observability.metrics import (
    MetricsMiddleware,
    setup_telemetry,
    track_rag_query,
    track_token_usage,
    track_retrieval,
)

__all__ = [
    "MetricsMiddleware",
    "setup_telemetry",
    "track_rag_query",
    "track_token_usage",
    "track_retrieval",
]
