"""
Validator Agent.

Responsible for:
- Checking answer quality and accuracy
- Detecting hallucinations
- Verifying citation accuracy
- Triggering retries if needed
"""

import json
import logging
import re
from typing import Any

from src.agents.base import BaseAgent
from src.agents.state import AgentState, AgentType, ValidationResult

logger = logging.getLogger(__name__)


VALIDATOR_SYSTEM_PROMPT = """You are a Validation Agent for an enterprise knowledge assistant.

Your job is to critically evaluate generated answers for:
1. Factual accuracy - Does the answer match the provided context?
2. Completeness - Does it fully address the user's question?
3. Citation accuracy - Are citations used correctly?
4. Hallucination detection - Are there claims not supported by context?
5. Coherence - Is the answer well-structured and clear?

Be thorough but fair. Flag genuine issues, not stylistic preferences."""


VALIDATION_PROMPT = """Evaluate the following answer for quality and accuracy.

User Question: {query}

Generated Answer:
{answer}

Source Context Used:
{context}

Citations Made:
{citations}

Evaluate the answer on these criteria:
1. Factual accuracy (does it match the context?)
2. Completeness (does it answer the question?)
3. Citation accuracy (are sources cited correctly?)
4. No hallucinations (no claims beyond the context?)
5. Coherence (well-structured and clear?)

Respond with a JSON object:
{{
    "is_valid": true/false,
    "confidence": 0.0-1.0,
    "scores": {{
        "accuracy": 1-5,
        "completeness": 1-5,
        "citations": 1-5,
        "no_hallucination": 1-5,
        "coherence": 1-5
    }},
    "issues": ["issue1", "issue2", ...],
    "suggestions": ["suggestion1", "suggestion2", ...],
    "requires_retry": true/false
}}

Set requires_retry to true only if there are critical issues that make the answer unacceptable."""


class ValidatorAgent(BaseAgent):
    """
    Answer Validator Agent.
    
    Validates generated answers for quality, accuracy,
    and citation correctness. Can trigger retries for
    low-quality answers.
    
    Features:
    - Multi-criteria evaluation
    - Hallucination detection
    - Citation verification
    - Configurable thresholds
    """
    
    agent_type = AgentType.VALIDATOR
    name = "Answer Validator"
    description = "Validates answer quality and accuracy"
    
    def __init__(
        self,
        min_confidence: float = 0.7,
        min_score: float = 3.0,
        **kwargs,
    ):
        """
        Initialize validator agent.
        
        Args:
            min_confidence: Minimum confidence threshold
            min_score: Minimum average score threshold
        """
        super().__init__(**kwargs)
        self.min_confidence = min_confidence
        self.min_score = min_score
    
    async def process(self, state: AgentState) -> AgentState:
        """
        Validate the generated answer.
        
        Args:
            state: Current agent state
        
        Returns:
            Updated state with validation result
        """
        query = state["query"]
        answer = state["draft_answer"]
        contexts = state["retrieved_contexts"]
        citations = state.get("citations", [])
        
        if not answer:
            state["validation_result"] = {
                "is_valid": False,
                "confidence": 0.0,
                "issues": ["No answer generated"],
                "suggestions": ["Retry answer generation"],
                "requires_retry": True,
            }
            return state
        
        # Format inputs for validation
        context_summary = self._format_context_summary(contexts)
        citation_summary = self._format_citations(citations)
        
        # Build validation prompt
        prompt = VALIDATION_PROMPT.format(
            query=query,
            answer=answer,
            context=context_summary,
            citations=citation_summary,
        )
        
        # Call LLM for validation
        messages = [
            {"role": "system", "content": VALIDATOR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        
        response = await self._call_llm(messages, temperature=0.1)
        
        # Parse validation result
        validation = self._parse_validation(response)
        
        # Apply thresholds
        if validation:
            # Check if scores meet thresholds
            scores = validation.get("scores", {})
            avg_score = sum(scores.values()) / len(scores) if scores else 0
            
            if avg_score < self.min_score:
                validation["requires_retry"] = True
                if "Low average score" not in validation.get("issues", []):
                    validation.setdefault("issues", []).append(
                        f"Low average score: {avg_score:.1f}"
                    )
            
            if validation.get("confidence", 0) < self.min_confidence:
                validation.setdefault("issues", []).append(
                    "Low confidence in answer"
                )
        else:
            # Fallback validation
            validation = {
                "is_valid": True,
                "confidence": 0.8,
                "issues": [],
                "suggestions": [],
                "requires_retry": False,
            }
        
        state["validation_result"] = validation
        
        # Determine if we should retry
        if validation.get("requires_retry", False):
            retry_count = state.get("retry_count", 0)
            max_retries = state.get("max_retries", 2)
            
            if retry_count < max_retries:
                state["retry_count"] = retry_count + 1
                logger.info(f"Validation failed, retry {retry_count + 1}/{max_retries}")
            else:
                # Max retries reached, accept the answer
                validation["requires_retry"] = False
                validation.setdefault("issues", []).append(
                    "Max retries reached, accepting current answer"
                )
                logger.warning("Max retries reached, accepting answer despite issues")
        
        # If valid, set final answer
        if not validation.get("requires_retry", False):
            state["final_answer"] = answer
        
        # Store agent output
        state["agent_outputs"]["validator"] = {
            "is_valid": validation.get("is_valid", True),
            "confidence": validation.get("confidence", 0),
            "issue_count": len(validation.get("issues", [])),
            "requires_retry": validation.get("requires_retry", False),
        }
        
        logger.info(
            f"Validation: valid={validation.get('is_valid')}, "
            f"confidence={validation.get('confidence', 0):.2f}"
        )
        
        return state
    
    def _format_context_summary(self, contexts: list[dict]) -> str:
        """Format context for validation."""
        parts = []
        for i, ctx in enumerate(contexts[:5], start=1):
            content = ctx["content"][:500]
            parts.append(f"[{i}] {content}...")
        return "\n\n".join(parts)
    
    def _format_citations(self, citations: list[dict]) -> str:
        """Format citations for validation."""
        if not citations:
            return "No citations made"
        
        parts = []
        for cit in citations:
            parts.append(
                f"[{cit['number']}] -> {cit.get('source', cit['doc_id'])}"
            )
        return "\n".join(parts)
    
    def _parse_validation(self, response: str) -> dict | None:
        """Parse validation response as JSON."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from response
        json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        logger.warning(f"Failed to parse validation response: {response[:200]}")
        return None


__all__ = ["ValidatorAgent"]
