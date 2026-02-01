"""
Tests for the multi-agent system.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.agents.state import (
    QueryType,
    AgentType,
    StepStatus,
    AgentState,
    create_initial_state,
)
from src.agents.base import BaseAgent
from src.agents.planner import PlannerAgent
from src.agents.qa import QAAgent
from src.agents.validator import ValidatorAgent


class TestAgentState:
    """Tests for agent state management."""
    
    def test_create_initial_state(self):
        """Test initial state creation."""
        state = create_initial_state(
            query="What is the return policy?",
            user_id="user123",
            max_retries=3,
        )
        
        assert state["query"] == "What is the return policy?"
        assert state["user_id"] == "user123"
        assert state["max_retries"] == 3
        assert state["retry_count"] == 0
        assert state["query_id"] is not None
        assert len(state["sub_queries"]) == 0
        assert len(state["errors"]) == 0
    
    def test_query_type_enum(self):
        """Test query type enumeration."""
        assert QueryType.FACTUAL.value == "factual"
        assert QueryType.MULTI_HOP.value == "multi_hop"
        assert QueryType.ANALYTICAL.value == "analytical"
    
    def test_step_status_enum(self):
        """Test step status enumeration."""
        assert StepStatus.PENDING.value == "pending"
        assert StepStatus.IN_PROGRESS.value == "in_progress"
        assert StepStatus.COMPLETED.value == "completed"


class TestPlannerAgent:
    """Tests for planner agent."""
    
    @pytest.mark.asyncio
    async def test_planner_fallback(self):
        """Test planner fallback when LLM fails."""
        planner = PlannerAgent()
        
        # Mock LLM to return non-JSON
        with patch.object(planner, '_call_llm', return_value="Unable to analyze"):
            state = create_initial_state(query="What is the policy?")
            result = await planner.process(state)
            
            # Should use fallback
            assert result["query_type"] == QueryType.FACTUAL.value
            assert len(result["sub_queries"]) == 1
    
    def test_entity_extraction_simple(self):
        """Test simple entity extraction."""
        planner = PlannerAgent()
        
        entities = planner._extract_entities_simple(
            'What is the policy at "Acme Corp" regarding John Smith?'
        )
        
        assert "Acme Corp" in entities
        assert "John Smith" in entities
    
    @pytest.mark.asyncio
    async def test_planner_with_valid_json(self):
        """Test planner with valid LLM response."""
        planner = PlannerAgent()
        
        mock_response = '''
        {
            "query_type": "multi_hop",
            "complexity": 3,
            "requires_multi_hop": true,
            "key_entities": ["vacation", "policy"],
            "sub_queries": [
                {
                    "id": "sq1",
                    "query": "What is the vacation policy?",
                    "type": "factual",
                    "priority": 1,
                    "depends_on": []
                }
            ]
        }
        '''
        
        with patch.object(planner, '_call_llm', return_value=mock_response):
            state = create_initial_state(query="Test query")
            result = await planner.process(state)
            
            assert result["query_type"] == "multi_hop"
            assert result["query_complexity"] == 3
            assert len(result["sub_queries"]) == 1


class TestQAAgent:
    """Tests for QA agent."""
    
    def test_format_context(self):
        """Test context formatting."""
        qa = QAAgent(max_context_length=500)
        
        contexts = [
            {
                "content": "This is document one content.",
                "doc_id": "doc1",
                "metadata": {"filename": "file1.pdf"},
            },
            {
                "content": "This is document two content.",
                "doc_id": "doc2",
                "metadata": {"filename": "file2.pdf"},
            },
        ]
        
        formatted = qa._format_context(contexts)
        
        assert "[1]" in formatted
        assert "[2]" in formatted
        assert "file1.pdf" in formatted
    
    def test_extract_citations(self):
        """Test citation extraction."""
        qa = QAAgent()
        
        answer = "According to [1], the policy allows 20 days. This is confirmed by [2] and [1]."
        contexts = [
            {"chunk_id": "c1", "doc_id": "d1", "content": "20 days", "metadata": {}},
            {"chunk_id": "c2", "doc_id": "d2", "content": "confirmed", "metadata": {}},
        ]
        
        citations = qa._extract_citations(answer, contexts)
        
        assert len(citations) == 2
        assert citations[0]["number"] == 1
        assert citations[0]["chunk_id"] == "c1"


class TestValidatorAgent:
    """Tests for validator agent."""
    
    def test_format_context_summary(self):
        """Test context summary formatting."""
        validator = ValidatorAgent()
        
        contexts = [
            {"content": "A" * 600, "doc_id": "d1"},
            {"content": "B" * 600, "doc_id": "d2"},
        ]
        
        summary = validator._format_context_summary(contexts)
        
        assert "[1]" in summary
        assert "[2]" in summary
    
    @pytest.mark.asyncio
    async def test_validator_with_valid_json(self):
        """Test validator with valid LLM response."""
        validator = ValidatorAgent()
        
        mock_response = '''
        {
            "is_valid": true,
            "confidence": 0.9,
            "scores": {
                "accuracy": 4,
                "completeness": 5,
                "citations": 4,
                "no_hallucination": 5,
                "coherence": 4
            },
            "issues": [],
            "suggestions": [],
            "requires_retry": false
        }
        '''
        
        with patch.object(validator, '_call_llm', return_value=mock_response):
            state = create_initial_state(query="Test")
            state["draft_answer"] = "This is an answer."
            state["retrieved_contexts"] = [
                {"content": "Test", "chunk_id": "c1", "doc_id": "d1"}
            ]
            
            result = await validator.process(state)
            
            assert result["validation_result"]["is_valid"] is True
            assert result["validation_result"]["confidence"] == 0.9
            assert result["final_answer"] == "This is an answer."
    
    @pytest.mark.asyncio
    async def test_validator_triggers_retry(self):
        """Test validator triggers retry for low quality answer."""
        validator = ValidatorAgent(min_score=4.0)
        
        mock_response = '''
        {
            "is_valid": false,
            "confidence": 0.3,
            "scores": {
                "accuracy": 2,
                "completeness": 2,
                "citations": 2,
                "no_hallucination": 2,
                "coherence": 2
            },
            "issues": ["Low accuracy", "Missing citations"],
            "suggestions": ["Improve accuracy"],
            "requires_retry": true
        }
        '''
        
        with patch.object(validator, '_call_llm', return_value=mock_response):
            state = create_initial_state(query="Test", max_retries=2)
            state["draft_answer"] = "Bad answer."
            state["retrieved_contexts"] = [
                {"content": "Test", "chunk_id": "c1", "doc_id": "d1"}
            ]
            
            result = await validator.process(state)
            
            assert result["validation_result"]["requires_retry"] is True
            assert result["retry_count"] == 1
            assert "final_answer" not in result or result["final_answer"] == ""


class TestWorkflow:
    """Tests for workflow orchestration."""
    
    def test_workflow_creation(self):
        """Test workflow creation."""
        from src.agents.workflow import RAGWorkflow, create_rag_workflow
        
        workflow = create_rag_workflow(max_retries=3)
        
        assert workflow.planner is not None
        assert workflow.extractor is not None
        assert workflow.qa is not None
        assert workflow.validator is not None
        assert workflow.max_retries == 3
    
    def test_workflow_builder(self):
        """Test workflow builder pattern."""
        from src.agents.workflow import RAGWorkflowBuilder
        
        custom_planner = PlannerAgent()
        
        workflow = (
            RAGWorkflowBuilder()
            .with_planner(custom_planner)
            .with_max_retries(5)
            .build()
        )
        
        assert workflow.planner is custom_planner
        assert workflow.max_retries == 5
