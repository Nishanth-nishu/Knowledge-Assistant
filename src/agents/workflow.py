"""
LangGraph Workflow Orchestration.

Defines the multi-agent workflow using LangGraph:
1. Planner -> analyzes query, creates sub-queries
2. Extractor -> retrieves context for each sub-query
3. QA -> synthesizes answer from context
4. Validator -> checks answer quality
5. Conditional retry loop if validation fails
"""

import logging
from datetime import datetime
from typing import Annotated, Any, Literal

from langgraph.constants import Send
from langgraph.graph import StateGraph, END

from src.agents.state import AgentState, create_initial_state
from src.agents.planner import PlannerAgent
from src.agents.extractor import ExtractorAgent
from src.agents.qa import QAAgent
from src.agents.validator import ValidatorAgent

logger = logging.getLogger(__name__)


# Type alias for routing decisions
RouteDecision = Literal["extract", "qa", "validate", "end", "retry"]


def should_retry(state: AgentState) -> RouteDecision:
    """
    Determine if we should retry based on validation result.
    
    Args:
        state: Current agent state
    
    Returns:
        Next node to execute
    """
    validation = state.get("validation_result")
    
    if validation and validation.get("requires_retry", False):
        return "retry"
    
    return "end"


def route_after_planning(state: AgentState) -> RouteDecision:
    """
    Route after planning step.
    
    Args:
        state: Current agent state
    
    Returns:
        Next node ("extract" always for now)
    """
    # Could add logic here to skip extraction for certain query types
    return "extract"


class RAGWorkflow:
    """
    Multi-agent RAG workflow using LangGraph.
    
    Orchestrates the flow:
    Planner -> Extractor -> QA -> Validator -> (retry or end)
    
    Example:
        workflow = RAGWorkflow()
        result = await workflow.run("What is the company's vacation policy?")
        print(result["final_answer"])
    """
    
    def __init__(
        self,
        planner: PlannerAgent | None = None,
        extractor: ExtractorAgent | None = None,
        qa: QAAgent | None = None,
        validator: ValidatorAgent | None = None,
        max_retries: int = 2,
    ):
        """
        Initialize RAG workflow.
        
        Args:
            planner: Planner agent instance
            extractor: Extractor agent instance
            qa: QA agent instance
            validator: Validator agent instance
            max_retries: Maximum validation retries
        """
        self.planner = planner or PlannerAgent()
        self.extractor = extractor or ExtractorAgent()
        self.qa = qa or QAAgent()
        self.validator = validator or ValidatorAgent()
        self.max_retries = max_retries
        
        self._graph = None
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create graph with our state schema
        graph = StateGraph(AgentState)
        
        # Add nodes for each agent
        graph.add_node("plan", self._plan_node)
        graph.add_node("extract", self._extract_node)
        graph.add_node("answer", self._answer_node)
        graph.add_node("validate", self._validate_node)
        graph.add_node("retry", self._retry_node)
        
        # Set entry point
        graph.set_entry_point("plan")
        
        # Add edges
        graph.add_edge("plan", "extract")
        graph.add_edge("extract", "answer")
        graph.add_edge("answer", "validate")
        
        # Conditional edge after validation
        graph.add_conditional_edges(
            "validate",
            should_retry,
            {
                "retry": "retry",
                "end": END,
            }
        )
        
        # Retry goes back to extract
        graph.add_edge("retry", "extract")
        
        return graph
    
    async def _plan_node(self, state: AgentState) -> AgentState:
        """Execute planner agent."""
        return await self.planner(state)
    
    async def _extract_node(self, state: AgentState) -> AgentState:
        """Execute extractor agent."""
        return await self.extractor(state)
    
    async def _answer_node(self, state: AgentState) -> AgentState:
        """Execute QA agent."""
        return await self.qa(state)
    
    async def _validate_node(self, state: AgentState) -> AgentState:
        """Execute validator agent."""
        return await self.validator(state)
    
    async def _retry_node(self, state: AgentState) -> AgentState:
        """
        Handle retry logic.
        
        Clears draft answer and prepares for re-extraction/generation.
        """
        state["draft_answer"] = ""
        state["validation_result"] = None
        
        # Add feedback from validation to improve next attempt
        validation = state.get("agent_outputs", {}).get("validator", {})
        issues = validation.get("issues", [])
        
        if issues:
            logger.info(f"Retrying due to: {issues}")
        
        return state
    
    @property
    def graph(self) -> StateGraph:
        """Get or build the workflow graph."""
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph
    
    def compile(self):
        """Compile the workflow for execution."""
        return self.graph.compile()
    
    async def run(
        self,
        query: str,
        user_id: str | None = None,
        session_id: str | None = None,
        config: dict | None = None,
    ) -> AgentState:
        """
        Run the RAG workflow on a query.
        
        Args:
            query: User's question
            user_id: Optional user ID
            session_id: Optional session ID
            config: Optional LangGraph config
        
        Returns:
            Final agent state with answer
        """
        # Create initial state
        state = create_initial_state(
            query=query,
            user_id=user_id,
            session_id=session_id,
            max_retries=self.max_retries,
        )
        
        # Compile and run
        app = self.compile()
        
        logger.info(f"Starting RAG workflow for query: {query[:50]}...")
        
        try:
            final_state = await app.ainvoke(state, config=config or {})
            
            # Set end time
            final_state["end_time"] = datetime.utcnow().isoformat()
            
            logger.info(
                f"Workflow completed: answer_length={len(final_state.get('final_answer', ''))}"
            )
            
            return final_state
            
        except Exception as e:
            logger.error(f"Workflow error: {e}")
            state["errors"].append(str(e))
            state["final_answer"] = "An error occurred while processing your question."
            return state
    
    async def stream(
        self,
        query: str,
        user_id: str | None = None,
        session_id: str | None = None,
    ):
        """
        Stream workflow execution events.
        
        Args:
            query: User's question
            user_id: Optional user ID
            session_id: Optional session ID
        
        Yields:
            State updates as workflow progresses
        """
        state = create_initial_state(
            query=query,
            user_id=user_id,
            session_id=session_id,
            max_retries=self.max_retries,
        )
        
        app = self.compile()
        
        async for event in app.astream(state):
            yield event


class RAGWorkflowBuilder:
    """Builder for customizing RAG workflow."""
    
    def __init__(self):
        self._planner = None
        self._extractor = None
        self._qa = None
        self._validator = None
        self._max_retries = 2
    
    def with_planner(self, planner: PlannerAgent) -> "RAGWorkflowBuilder":
        """Set custom planner agent."""
        self._planner = planner
        return self
    
    def with_extractor(self, extractor: ExtractorAgent) -> "RAGWorkflowBuilder":
        """Set custom extractor agent."""
        self._extractor = extractor
        return self
    
    def with_qa(self, qa: QAAgent) -> "RAGWorkflowBuilder":
        """Set custom QA agent."""
        self._qa = qa
        return self
    
    def with_validator(self, validator: ValidatorAgent) -> "RAGWorkflowBuilder":
        """Set custom validator agent."""
        self._validator = validator
        return self
    
    def with_max_retries(self, max_retries: int) -> "RAGWorkflowBuilder":
        """Set max retries."""
        self._max_retries = max_retries
        return self
    
    def build(self) -> RAGWorkflow:
        """Build the workflow."""
        return RAGWorkflow(
            planner=self._planner,
            extractor=self._extractor,
            qa=self._qa,
            validator=self._validator,
            max_retries=self._max_retries,
        )


# Factory function
def create_rag_workflow(
    max_retries: int = 2,
    **agent_kwargs,
) -> RAGWorkflow:
    """
    Create a RAG workflow with default agents.
    
    Args:
        max_retries: Maximum validation retries
        **agent_kwargs: Additional agent configuration
    
    Returns:
        Configured RAGWorkflow instance
    """
    return RAGWorkflow(
        planner=PlannerAgent(**agent_kwargs),
        extractor=ExtractorAgent(**agent_kwargs),
        qa=QAAgent(**agent_kwargs),
        validator=ValidatorAgent(**agent_kwargs),
        max_retries=max_retries,
    )


__all__ = [
    "RAGWorkflow",
    "RAGWorkflowBuilder",
    "create_rag_workflow",
]
