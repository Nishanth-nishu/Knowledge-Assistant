"""
Query endpoints for the RAG system.

Now integrated with the multi-agent RAG workflow and protected by authentication.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.api.middleware.auth import UserClaims, require_auth, require_scope
from src.agents.workflow import RAGWorkflow, create_rag_workflow
from src.agents.state import AgentState

logger = logging.getLogger(__name__)

router = APIRouter()

# Global workflow instance (lazy initialized)
_workflow: RAGWorkflow | None = None


def get_workflow() -> RAGWorkflow:
    """Get or create RAG workflow instance."""
    global _workflow
    if _workflow is None:
        _workflow = create_rag_workflow(max_retries=2)
    return _workflow


# =============================================================================
# Request/Response Models
# =============================================================================

class QueryRequest(BaseModel):
    """Query request model."""
    query: str = Field(..., min_length=1, max_length=10000, description="The user's query")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    use_reranker: bool = Field(default=True, description="Whether to use ColBERT reranking")
    use_knowledge_graph: bool = Field(default=True, description="Whether to include KG results")
    include_sources: bool = Field(default=True, description="Include source documents in response")
    stream: bool = Field(default=False, description="Stream the response")
    
    # Optional filters
    document_ids: list[str] | None = Field(default=None, description="Filter by document IDs")
    metadata_filter: dict[str, Any] | None = Field(default=None, description="Metadata filters")


class Source(BaseModel):
    """Source document reference."""
    document_id: str
    chunk_id: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    page_number: int | None = None


class Citation(BaseModel):
    """Citation reference."""
    number: int
    chunk_id: str
    source: str
    snippet: str


class QueryResponse(BaseModel):
    """Query response model."""
    query_id: str
    query: str
    answer: str
    sources: list[Source] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_steps: list[str] = Field(default_factory=list)
    
    # Metadata
    latency_ms: float
    tokens_used: int
    model: str
    timestamp: str
    
    # Agent info
    query_type: str | None = None
    sub_queries_count: int = 0


class StreamChunk(BaseModel):
    """Streaming response chunk."""
    type: str  # "status", "content", "source", "done", "error"
    content: str | None = None
    source: Source | None = None
    agent: str | None = None


# Query result cache (replace with Redis in production)
_query_cache: dict[str, QueryResponse] = {}


# =============================================================================
# Endpoints
# =============================================================================

@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query the Knowledge Base",
    description="""
    Submit a query to the enterprise knowledge assistant.
    
    The system will:
    1. Analyze the query using the Planner Agent
    2. Retrieve relevant documents using hybrid search (Extractor Agent)
    3. Rerank results using ColBERT
    4. Generate an answer with citations (QA Agent)
    5. Validate the response for accuracy (Validator Agent)
    """,
)
async def query(
    request: QueryRequest,
    user: UserClaims = Depends(require_auth),
) -> QueryResponse:
    """
    Process a query through the RAG pipeline.
    
    Requires authentication. Rate limited per user.
    """
    start_time = time.time()
    query_id = str(uuid.uuid4())
    
    logger.info(f"Query {query_id} from user {user.user_id}: {request.query[:50]}...")
    
    try:
        # Get workflow
        workflow = get_workflow()
        
        # Configure extractor based on request
        workflow.extractor.top_k = request.top_k
        workflow.extractor.use_reranker = request.use_reranker
        workflow.extractor.use_knowledge_graph = request.use_knowledge_graph
        
        # Run workflow
        state = await workflow.run(
            query=request.query,
            user_id=user.user_id,
            session_id=query_id,
        )
        
        # Extract results
        latency_ms = (time.time() - start_time) * 1000
        
        # Build sources
        sources = []
        if request.include_sources:
            for ctx in state.get("retrieved_contexts", [])[:request.top_k]:
                sources.append(Source(
                    document_id=ctx.get("doc_id", ""),
                    chunk_id=ctx.get("chunk_id", ""),
                    content=ctx.get("content", "")[:500],
                    score=ctx.get("score", 0.0),
                    metadata=ctx.get("metadata", {}),
                ))
        
        # Build citations
        citations = [
            Citation(
                number=c.get("number", 0),
                chunk_id=c.get("chunk_id", ""),
                source=c.get("source", ""),
                snippet=c.get("snippet", "")[:200],
            )
            for c in state.get("citations", [])
        ]
        
        # Get validation confidence
        validation = state.get("validation_result", {})
        confidence = validation.get("confidence", 0.8) if validation else 0.8
        
        response = QueryResponse(
            query_id=query_id,
            query=request.query,
            answer=state.get("final_answer", ""),
            sources=sources,
            citations=citations,
            confidence=confidence,
            reasoning_steps=[
                f"Query analyzed: {state.get('query_type', 'unknown')}",
                f"Sub-queries: {len(state.get('sub_queries', []))}",
                f"Contexts retrieved: {len(state.get('retrieved_contexts', []))}",
                f"Answer generated and validated",
            ],
            latency_ms=latency_ms,
            tokens_used=0,  # TODO: Track token usage
            model="multi-agent-rag",
            timestamp=datetime.utcnow().isoformat(),
            query_type=state.get("query_type"),
            sub_queries_count=len(state.get("sub_queries", [])),
        )
        
        # Cache result
        _query_cache[query_id] = response
        
        logger.info(f"Query {query_id} completed in {latency_ms:.0f}ms")
        
        return response
        
    except Exception as e:
        logger.error(f"Query {query_id} failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}",
        )


@router.post(
    "/query/stream",
    summary="Query with Streaming Response",
    description="Submit a query and receive a streaming response with real-time updates.",
)
async def query_stream(
    request: QueryRequest,
    user: UserClaims = Depends(require_auth),
):
    """
    Process a query with streaming response.
    
    Returns Server-Sent Events with status updates and content chunks.
    """
    query_id = str(uuid.uuid4())
    
    async def generate() -> AsyncGenerator[str, None]:
        try:
            workflow = get_workflow()
            
            # Configure workflow
            workflow.extractor.top_k = request.top_k
            workflow.extractor.use_reranker = request.use_reranker
            
            # Status update
            yield json.dumps({
                "type": "status",
                "content": "Starting query processing...",
                "agent": "orchestrator",
            }) + "\n"
            
            # Run workflow with streaming
            async for event in workflow.stream(
                query=request.query,
                user_id=user.user_id,
                session_id=query_id,
            ):
                # Send status updates for each agent
                for key, value in event.items():
                    if isinstance(value, dict) and "current_agent" in value:
                        agent = value.get("current_agent", "")
                        yield json.dumps({
                            "type": "status",
                            "content": f"Processing with {agent}...",
                            "agent": agent,
                        }) + "\n"
                    
                    # Stream answer chunks
                    if key == "answer" and value.get("draft_answer"):
                        yield json.dumps({
                            "type": "content",
                            "content": value.get("draft_answer", ""),
                        }) + "\n"
            
            # Final result
            yield json.dumps({
                "type": "done",
                "query_id": query_id,
            }) + "\n"
            
        except Exception as e:
            logger.error(f"Streaming query {query_id} failed: {e}")
            yield json.dumps({
                "type": "error",
                "content": str(e),
            }) + "\n"
    
    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Query-ID": query_id,
        },
    )


@router.get(
    "/query/{query_id}",
    response_model=QueryResponse,
    summary="Get Query Result",
    description="Retrieve a previously executed query result by ID.",
)
async def get_query(
    query_id: str,
    user: UserClaims = Depends(require_auth),
) -> QueryResponse:
    """
    Get a query result by ID.
    
    Results are cached for a configurable period.
    """
    if query_id in _query_cache:
        return _query_cache[query_id]
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Query {query_id} not found or expired",
    )


@router.delete(
    "/query/{query_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Query Result",
)
async def delete_query(
    query_id: str,
    user: UserClaims = Depends(require_auth),
):
    """Delete a cached query result."""
    if query_id in _query_cache:
        del _query_cache[query_id]
    
    return None


__all__ = ["router"]
