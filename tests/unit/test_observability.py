"""
Tests for observability components.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.observability.metrics import (
    MetricsMiddleware,
    track_rag_query,
    track_token_usage,
    track_retrieval,
)


class TestMetrics:
    """Tests for metrics collection."""
    
    @patch("src.observability.metrics.RAG_QUERY_TOTAL")
    def test_track_rag_query(self, mock_counter):
        """Test tracking rag queries."""
        track_rag_query("success", "planner")
        
        mock_counter.labels.assert_called_with(status="success", agent_type="planner")
        mock_counter.labels.return_value.inc.assert_called()
    
    @patch("src.observability.metrics.RAG_TOKEN_USAGE")
    def test_track_token_usage(self, mock_counter):
        """Test tracking token usage."""
        track_token_usage("gpt-4", 100, 50)
        
        # Should be called twice (input and output)
        assert mock_counter.labels.call_count == 2
        mock_counter.labels.return_value.inc.assert_called()
    
    @patch("src.observability.metrics.RAG_RETRIEVAL_COUNT")
    def test_track_retrieval(self, mock_hist):
        """Test tracking retrieval."""
        track_retrieval("hybrid", 10)
        
        mock_hist.labels.assert_called_with(method="hybrid")
        mock_hist.labels.return_value.observe.assert_called_with(10)


@pytest.mark.asyncio
async def test_metrics_middleware():
    """Test metrics middleware."""
    app = MagicMock()
    app.return_value = None  # Mock awaitable
    
    middleware = MetricsMiddleware(app)
    
    # Mock ASGI scope, receive, send
    scope = {"type": "http", "method": "GET", "path": "/test"}
    receive = AsyncMock()
    send = AsyncMock()
    
    await middleware(scope, receive, send)
    
    app.assert_called_once()
    
    # helper for async mock
    async def AsyncMock(*args, **kwargs):
        pass
