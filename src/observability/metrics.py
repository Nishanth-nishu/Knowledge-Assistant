"""
Metrics and Observability.

Provides:
- Prometheus metrics instrumentation
- OpenTelemetry tracing setup
- Custom metrics for RAG pipeline
"""

import time
import logging
from typing import Callable

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from fastapi import Request, Response
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource

from src.config import get_settings

logger = logging.getLogger(__name__)

# =============================================================================
# Prometheus Metrics
# =============================================================================

# HTTP Metrics
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

# RAG Pipeline Metrics
RAG_QUERY_TOTAL = Counter(
    "rag_query_total",
    "Total RAG queries processed",
    ["status", "agent_type"],
)

RAG_QUERY_DURATION = Histogram(
    "rag_query_duration_seconds",
    "RAG query processing duration",
    ["step"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

RAG_RETRIEVAL_COUNT = Histogram(
    "rag_retrieval_count",
    "Number of documents retrieved",
    ["method"],
    buckets=[0, 1, 5, 10, 20, 50, 100],
)

RAG_TOKEN_USAGE = Counter(
    "rag_token_usage_total",
    "Total LLM tokens used",
    ["model", "type"],  # type: input, output
)

# System Metrics
SYSTEM_INFO = Gauge(
    "system_info",
    "System information",
    ["version", "environment"],
)


class MetricsMiddleware:
    """Middleware for collecting HTTP metrics."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        
        start_time = time.time()
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_code = str(message["status"])
                method = scope["method"]
                path = scope["path"]
                
                # Update metrics
                duration = time.time() - start_time
                HTTP_REQUESTS_TOTAL.labels(method, path, status_code).inc()
                HTTP_REQUEST_DURATION.labels(method, path).observe(duration)
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)


# =============================================================================
# Tracing Setup
# =============================================================================

def setup_telemetry(service_name: str = "eka-service"):
    """
    Configure OpenTelemetry tracing.
    
    Args:
        service_name: Name of the service
    """
    settings = get_settings()
    
    if not settings.monitoring.enable_tracing:
        return
    
    resource = Resource.create({
        "service.name": service_name,
        "service.version": "0.1.0",
        "deployment.environment": settings.environment,
    })
    
    tracer_provider = TracerProvider(resource=resource)
    
    # OTLP Exporter (sends to Jaeger/Tempo)
    otlp_exporter = OTLPSpanExporter(
        endpoint=settings.monitoring.otlp_endpoint,
        insecure=True,
    )
    
    span_processor = BatchSpanProcessor(otlp_exporter)
    tracer_provider.add_span_processor(span_processor)
    
    trace.set_tracer_provider(tracer_provider)
    
    logger.info(f"OpenTelemetry tracing enabled for {service_name}")


# =============================================================================
# Helper Functions
# =============================================================================

def track_rag_query(status: str, agent_type: str = "orchestrator"):
    """Track a RAG query event."""
    RAG_QUERY_TOTAL.labels(status=status, agent_type=agent_type).inc()

def track_token_usage(model: str, input_tokens: int, output_tokens: int):
    """Track token usage."""
    RAG_TOKEN_USAGE.labels(model=model, type="input").inc(input_tokens)
    RAG_TOKEN_USAGE.labels(model=model, type="output").inc(output_tokens)

def track_retrieval(method: str, count: int):
    """Track retrieval stats."""
    RAG_RETRIEVAL_COUNT.labels(method=method).observe(count)


__all__ = [
    "MetricsMiddleware",
    "setup_telemetry",
    "track_rag_query",
    "track_token_usage",
    "track_retrieval",
    "HTTP_REQUESTS_TOTAL",
    "RAG_QUERY_DURATION",
]
