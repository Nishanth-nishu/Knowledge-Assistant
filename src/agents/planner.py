"""
Planner Agent.

Responsible for:
- Analyzing user queries
- Determining query type and complexity
- Decomposing complex queries into sub-queries
- Planning the retrieval strategy
"""

import json
import logging
import re
from typing import Any

from src.agents.base import BaseAgent
from src.agents.state import (
    AgentState,
    AgentType,
    QueryType,
    StepStatus,
    SubQuery,
)

logger = logging.getLogger(__name__)


PLANNER_SYSTEM_PROMPT = """You are a Query Planner for an enterprise knowledge assistant.

Your job is to:
1. Analyze the user's query to understand its intent and complexity
2. Classify the query type (factual, analytical, comparative, procedural, multi_hop)
3. Identify key entities and concepts mentioned
4. If the query is complex, decompose it into simpler sub-queries

For query decomposition:
- Break down multi-hop questions into sequential steps
- Identify dependencies between sub-queries
- Order sub-queries by dependency (what needs to be answered first)

Always respond in valid JSON format."""


QUERY_ANALYSIS_PROMPT = """Analyze the following user query and provide your analysis.

User Query: {query}

Respond with a JSON object containing:
{{
    "query_type": "factual|analytical|comparative|procedural|multi_hop",
    "complexity": 1-5 (1=simple, 5=very complex),
    "requires_multi_hop": true/false,
    "key_entities": ["entity1", "entity2", ...],
    "sub_queries": [
        {{
            "id": "sq1",
            "query": "The sub-query text",
            "type": "factual|analytical|...",
            "priority": 1 (execution order),
            "depends_on": [] (list of sub-query IDs this depends on)
        }},
        ...
    ]
}}

If the query is simple and doesn't need decomposition, return an empty sub_queries array.
Only decompose if the query genuinely requires multiple information lookups or reasoning steps."""


class PlannerAgent(BaseAgent):
    """
    Query Planner Agent.
    
    Analyzes incoming queries and creates an execution plan.
    For complex queries, decomposes them into sub-queries
    with dependency tracking.
    """
    
    agent_type = AgentType.PLANNER
    name = "Query Planner"
    description = "Analyzes queries and creates execution plans"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = PLANNER_SYSTEM_PROMPT
    
    async def process(self, state: AgentState) -> AgentState:
        """
        Analyze query and create execution plan.
        
        Args:
            state: Current agent state
        
        Returns:
            Updated state with query analysis and sub-queries
        """
        query = state["query"]
        
        # Call LLM for analysis
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": QUERY_ANALYSIS_PROMPT.format(query=query)},
        ]
        
        response = await self._call_llm(messages, temperature=0.1)
        
        # Parse response
        analysis = self._parse_analysis(response)
        
        if analysis:
            state["query_type"] = analysis.get("query_type", QueryType.FACTUAL.value)
            state["query_complexity"] = analysis.get("complexity", 1)
            state["requires_multi_hop"] = analysis.get("requires_multi_hop", False)
            state["key_entities"] = analysis.get("key_entities", [])
            
            # Process sub-queries
            sub_queries = []
            for sq in analysis.get("sub_queries", []):
                sub_query = {
                    "sub_query_id": sq.get("id", f"sq{len(sub_queries)}"),
                    "query": sq.get("query", ""),
                    "query_type": sq.get("type", QueryType.FACTUAL.value),
                    "priority": sq.get("priority", 1),
                    "depends_on": sq.get("depends_on", []),
                    "status": StepStatus.PENDING.value,
                    "answer": None,
                    "sources": [],
                }
                sub_queries.append(sub_query)
            
            # If no sub-queries, treat the main query as the only sub-query
            if not sub_queries:
                sub_queries = [{
                    "sub_query_id": "sq0",
                    "query": query,
                    "query_type": state["query_type"],
                    "priority": 1,
                    "depends_on": [],
                    "status": StepStatus.PENDING.value,
                    "answer": None,
                    "sources": [],
                }]
            
            state["sub_queries"] = sub_queries
            state["current_sub_query_idx"] = 0
        else:
            # Fallback: treat as simple factual query
            state["query_type"] = QueryType.FACTUAL.value
            state["query_complexity"] = 1
            state["requires_multi_hop"] = False
            state["key_entities"] = self._extract_entities_simple(query)
            state["sub_queries"] = [{
                "sub_query_id": "sq0",
                "query": query,
                "query_type": QueryType.FACTUAL.value,
                "priority": 1,
                "depends_on": [],
                "status": StepStatus.PENDING.value,
                "answer": None,
                "sources": [],
            }]
        
        # Store agent output
        state["agent_outputs"]["planner"] = {
            "query_type": state["query_type"],
            "complexity": state["query_complexity"],
            "sub_query_count": len(state["sub_queries"]),
            "key_entities": state["key_entities"],
        }
        
        logger.info(
            f"Query analyzed: type={state['query_type']}, "
            f"complexity={state['query_complexity']}, "
            f"sub_queries={len(state['sub_queries'])}"
        )
        
        return state
    
    def _parse_analysis(self, response: str) -> dict | None:
        """Parse LLM response as JSON."""
        try:
            # Try direct JSON parse
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object in response
        json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        logger.warning(f"Failed to parse planner response: {response[:200]}")
        return None
    
    def _extract_entities_simple(self, query: str) -> list[str]:
        """Simple entity extraction fallback."""
        # Extract quoted terms and capitalized words
        entities = []
        
        # Quoted strings
        quoted = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted)
        
        # Capitalized words (potential named entities)
        caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        entities.extend(caps)
        
        return list(set(entities))


__all__ = ["PlannerAgent"]
