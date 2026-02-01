"""
QA Agent.

Responsible for:
- Synthesizing answers from retrieved contexts
- Handling multi-hop reasoning
- Generating citations
- Producing coherent final answers
"""

import json
import logging
import re
from typing import Any

from src.agents.base import BaseAgent
from src.agents.state import AgentState, AgentType, StepStatus

logger = logging.getLogger(__name__)


QA_SYSTEM_PROMPT = """You are a Question Answering assistant for an enterprise knowledge system.

Your job is to:
1. Analyze the provided context carefully
2. Synthesize a comprehensive, accurate answer to the user's question
3. Cite your sources using [1], [2], etc. notation
4. If the context doesn't contain enough information, say so clearly
5. Be concise but thorough

Guidelines:
- Only use information from the provided context
- Always cite sources for factual claims
- If multiple sources agree, cite all of them
- If sources conflict, mention the discrepancy
- Structure longer answers with clear sections
- Use bullet points for lists when appropriate"""


ANSWER_PROMPT = """Based on the following context, answer the user's question.

User Question: {query}

Context Documents:
{context}

{kg_context}

Instructions:
1. Synthesize a clear, accurate answer using ONLY the provided context
2. Cite sources using [1], [2], etc. matching the document numbers above
3. If the context is insufficient, explicitly state what information is missing
4. Be concise but comprehensive

Your Answer:"""


class QAAgent(BaseAgent):
    """
    Question Answering Agent.
    
    Synthesizes answers from retrieved context using
    the LLM with citation support.
    
    Features:
    - Context-grounded answer generation
    - Automatic citation extraction
    - Multi-hop reasoning support
    - Structured answer formatting
    """
    
    agent_type = AgentType.QA
    name = "QA Synthesizer"
    description = "Synthesizes answers from retrieved context"
    
    def __init__(
        self,
        max_context_length: int = 8000,
        temperature: float = 0.3,
        **kwargs,
    ):
        """
        Initialize QA agent.
        
        Args:
            max_context_length: Maximum context characters
            temperature: LLM temperature for generation
        """
        super().__init__(**kwargs)
        self.max_context_length = max_context_length
        self.temperature = temperature
    
    async def process(self, state: AgentState) -> AgentState:
        """
        Generate answer from retrieved context.
        
        Args:
            state: Current agent state
        
        Returns:
            Updated state with draft answer
        """
        query = state["query"]
        contexts = state["retrieved_contexts"]
        kg_context = state.get("knowledge_graph_context")
        
        if not contexts:
            state["draft_answer"] = "I couldn't find relevant information to answer your question."
            state["citations"] = []
            return state
        
        # Format context for prompt
        formatted_context = self._format_context(contexts)
        
        # Format knowledge graph context if available
        kg_section = ""
        if kg_context:
            kg_section = self._format_kg_context(kg_context)
        
        # Build prompt
        prompt = ANSWER_PROMPT.format(
            query=query,
            context=formatted_context,
            kg_context=kg_section,
        )
        
        # Generate answer
        messages = [
            {"role": "system", "content": QA_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        
        response = await self._call_llm(
            messages,
            temperature=self.temperature,
            max_tokens=2000,
        )
        
        # Extract citations
        citations = self._extract_citations(response, contexts)
        
        # Store results
        state["draft_answer"] = response
        state["citations"] = citations
        
        # Mark sub-queries as completed
        for sq in state["sub_queries"]:
            if sq["status"] == StepStatus.IN_PROGRESS.value:
                sq["status"] = StepStatus.COMPLETED.value
                sq["answer"] = response
        
        # Store agent output
        state["agent_outputs"]["qa"] = {
            "answer_length": len(response),
            "citation_count": len(citations),
            "contexts_used": len(contexts),
        }
        
        logger.info(f"Generated answer with {len(citations)} citations")
        
        return state
    
    def _format_context(self, contexts: list[dict]) -> str:
        """
        Format contexts for the prompt.
        
        Args:
            contexts: Retrieved context dicts
        
        Returns:
            Formatted context string
        """
        formatted_parts = []
        total_length = 0
        
        for i, ctx in enumerate(contexts, start=1):
            content = ctx["content"]
            source = ctx.get("metadata", {}).get("filename", f"Document {ctx['doc_id']}")
            
            section = f"[{i}] Source: {source}\n{content}\n"
            
            # Check length limit
            if total_length + len(section) > self.max_context_length:
                # Truncate content
                available = self.max_context_length - total_length - 100
                if available > 200:
                    content = content[:available] + "..."
                    section = f"[{i}] Source: {source}\n{content}\n"
                else:
                    break
            
            formatted_parts.append(section)
            total_length += len(section)
        
        return "\n".join(formatted_parts)
    
    def _format_kg_context(self, kg_context: dict) -> str:
        """
        Format knowledge graph context.
        
        Args:
            kg_context: KG context dict
        
        Returns:
            Formatted KG context string
        """
        parts = []
        
        entities = kg_context.get("entities", [])
        if entities:
            entity_list = ", ".join(
                f"{e['name']} ({e['type']})" for e in entities[:5]
            )
            parts.append(f"Relevant Entities: {entity_list}")
        
        related = kg_context.get("related", [])
        if related:
            related_list = ", ".join(
                f"{r['name']} ({r['type']})" for r in related[:5]
            )
            parts.append(f"Related Concepts: {related_list}")
        
        if parts:
            return "\nKnowledge Graph Context:\n" + "\n".join(parts)
        
        return ""
    
    def _extract_citations(
        self,
        answer: str,
        contexts: list[dict],
    ) -> list[dict]:
        """
        Extract citation references from answer.
        
        Args:
            answer: Generated answer
            contexts: Original contexts
        
        Returns:
            List of citation dicts
        """
        citations = []
        
        # Find all citation markers like [1], [2], etc.
        citation_pattern = re.compile(r'\[(\d+)\]')
        matches = citation_pattern.findall(answer)
        
        # Get unique citation numbers
        cited_numbers = sorted(set(int(m) for m in matches))
        
        for num in cited_numbers:
            idx = num - 1  # Convert to 0-indexed
            if 0 <= idx < len(contexts):
                ctx = contexts[idx]
                citations.append({
                    "number": num,
                    "chunk_id": ctx["chunk_id"],
                    "doc_id": ctx["doc_id"],
                    "source": ctx.get("metadata", {}).get("filename", ""),
                    "snippet": ctx["content"][:200],
                })
        
        return citations


__all__ = ["QAAgent"]
