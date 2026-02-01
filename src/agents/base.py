"""
Base Agent Class.

Provides the foundation for all agents in the multi-agent system.
Each agent inherits from this class and implements specific behaviors.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from src.agents.state import AgentState, AgentType
from src.models.llm_client import LLMClient, get_llm_client

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Base class for all agents.
    
    Provides:
    - Access to LLM client
    - State management utilities
    - Logging and timing
    - Error handling
    """
    
    agent_type: AgentType
    name: str
    description: str
    
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        **kwargs,
    ):
        """
        Initialize base agent.
        
        Args:
            llm_client: LLM client instance
            **kwargs: Additional configuration
        """
        self._llm_client = llm_client
        self.config = kwargs
    
    @property
    def llm(self) -> LLMClient:
        """Get LLM client with lazy initialization."""
        if self._llm_client is None:
            self._llm_client = get_llm_client()
        return self._llm_client
    
    @abstractmethod
    async def process(self, state: AgentState) -> AgentState:
        """
        Process the state and return updated state.
        
        Args:
            state: Current agent state
        
        Returns:
            Updated agent state
        """
        pass
    
    async def __call__(self, state: AgentState) -> AgentState:
        """
        Execute the agent with timing and error handling.
        
        Args:
            state: Current agent state
        
        Returns:
            Updated agent state
        """
        start_time = time.time()
        state["current_agent"] = self.agent_type.value
        
        logger.info(f"Agent {self.name} starting processing")
        
        try:
            state = await self.process(state)
            
            # Record timing
            elapsed = time.time() - start_time
            state["step_timings"][self.agent_type.value] = elapsed
            
            logger.info(f"Agent {self.name} completed in {elapsed:.2f}s")
            
        except Exception as e:
            logger.error(f"Agent {self.name} error: {e}")
            state["errors"].append(f"{self.name}: {str(e)}")
        
        return state
    
    def _add_message(
        self,
        state: AgentState,
        role: str,
        content: str,
    ) -> None:
        """Add a message to the conversation history."""
        from datetime import datetime
        
        state["messages"].append({
            "role": role,
            "content": content,
            "agent": self.agent_type.value,
            "timestamp": datetime.utcnow().isoformat(),
        })
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        return f"You are the {self.name}. {self.description}"
    
    async def _call_llm(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Call the LLM with messages.
        
        Args:
            messages: List of message dicts
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
        
        Returns:
            LLM response content
        """
        response = await self.llm.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.content


class AgentRegistry:
    """Registry for agent instances."""
    
    _agents: dict[AgentType, BaseAgent] = {}
    
    @classmethod
    def register(cls, agent: BaseAgent) -> None:
        """Register an agent instance."""
        cls._agents[agent.agent_type] = agent
    
    @classmethod
    def get(cls, agent_type: AgentType) -> BaseAgent | None:
        """Get an agent by type."""
        return cls._agents.get(agent_type)
    
    @classmethod
    def all(cls) -> list[BaseAgent]:
        """Get all registered agents."""
        return list(cls._agents.values())
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered agents."""
        cls._agents.clear()


__all__ = [
    "BaseAgent",
    "AgentRegistry",
]
