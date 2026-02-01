"""
Agent State and Shared Types.

Defines the state schema and types used across all agents
in the multi-agent RAG system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TypedDict

from pydantic import BaseModel, Field


class QueryType(str, Enum):
    """Types of user queries."""
    FACTUAL = "factual"           # Direct fact lookup
    ANALYTICAL = "analytical"      # Requires analysis/reasoning
    COMPARATIVE = "comparative"    # Compare multiple entities
    PROCEDURAL = "procedural"      # How-to questions
    MULTI_HOP = "multi_hop"        # Requires chaining information
    CLARIFICATION = "clarification"  # Ambiguous, needs clarification


class AgentType(str, Enum):
    """Types of agents in the system."""
    PLANNER = "planner"
    EXTRACTOR = "extractor"
    QA = "qa"
    VALIDATOR = "validator"
    ORCHESTRATOR = "orchestrator"


class StepStatus(str, Enum):
    """Status of an execution step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class SubQuery:
    """A decomposed sub-query."""
    sub_query_id: str
    query: str
    query_type: QueryType
    priority: int = 1
    depends_on: list[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    answer: str | None = None
    sources: list[str] = field(default_factory=list)


@dataclass
class RetrievedContext:
    """Context retrieved for a query."""
    chunk_id: str
    content: str
    score: float
    source: str  # bm25, dense, sparse, kg
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of answer validation."""
    is_valid: bool
    confidence: float
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    requires_retry: bool = False


class Message(BaseModel):
    """A message in the agent conversation."""
    role: str = Field(..., description="Message role: user, assistant, system")
    content: str = Field(..., description="Message content")
    agent: str | None = Field(None, description="Agent that created this message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AgentState(TypedDict, total=False):
    """
    Shared state passed between agents in the LangGraph workflow.
    
    This state is modified by each agent as they process the query.
    """
    # Original input
    query: str
    query_id: str
    user_id: str | None
    session_id: str | None
    
    # Query analysis
    query_type: str
    query_complexity: int  # 1-5 scale
    requires_multi_hop: bool
    key_entities: list[str]
    
    # Query decomposition
    sub_queries: list[dict]  # List of SubQuery as dict
    current_sub_query_idx: int
    
    # Retrieval
    retrieved_contexts: list[dict]  # List of RetrievedContext as dict
    knowledge_graph_context: dict | None
    
    # Generation
    draft_answer: str
    final_answer: str
    sources: list[str]
    citations: list[dict]
    
    # Validation
    validation_result: dict | None  # ValidationResult as dict
    retry_count: int
    max_retries: int
    
    # Metadata
    messages: list[dict]  # Conversation history
    current_agent: str
    agent_outputs: dict[str, dict]  # Outputs from each agent
    errors: list[str]
    
    # Timing
    start_time: str
    end_time: str | None
    step_timings: dict[str, float]


def create_initial_state(
    query: str,
    query_id: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    max_retries: int = 2,
) -> AgentState:
    """
    Create initial agent state for a new query.
    
    Args:
        query: User's query
        query_id: Optional query ID
        user_id: Optional user ID
        session_id: Optional session ID
        max_retries: Maximum validation retries
    
    Returns:
        Initialized AgentState
    """
    import uuid
    
    return AgentState(
        query=query,
        query_id=query_id or str(uuid.uuid4()),
        user_id=user_id,
        session_id=session_id,
        query_type=QueryType.FACTUAL.value,
        query_complexity=1,
        requires_multi_hop=False,
        key_entities=[],
        sub_queries=[],
        current_sub_query_idx=0,
        retrieved_contexts=[],
        knowledge_graph_context=None,
        draft_answer="",
        final_answer="",
        sources=[],
        citations=[],
        validation_result=None,
        retry_count=0,
        max_retries=max_retries,
        messages=[],
        current_agent="",
        agent_outputs={},
        errors=[],
        start_time=datetime.utcnow().isoformat(),
        end_time=None,
        step_timings={},
    )


__all__ = [
    "QueryType",
    "AgentType",
    "StepStatus",
    "SubQuery",
    "RetrievedContext",
    "ValidationResult",
    "Message",
    "AgentState",
    "create_initial_state",
]
