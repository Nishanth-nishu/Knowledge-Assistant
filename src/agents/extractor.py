"""
Extractor Agent.

Responsible for:
- Retrieving relevant documents for each sub-query
- Using hybrid search (BM25 + Dense + Sparse)
- Enriching with knowledge graph context
- Reranking results for relevance
"""

import logging
from typing import Any

from src.agents.base import BaseAgent
from src.agents.state import AgentState, AgentType, StepStatus

logger = logging.getLogger(__name__)


class ExtractorAgent(BaseAgent):
    """
    Context Extractor Agent.
    
    Retrieves relevant information from the knowledge base
    using hybrid search and knowledge graph traversal.
    
    Features:
    - Three-way hybrid retrieval (BM25 + Dense + Sparse)
    - Knowledge graph context enrichment
    - Result reranking with ColBERT
    - Source deduplication
    """
    
    agent_type = AgentType.EXTRACTOR
    name = "Context Extractor"
    description = "Retrieves relevant context for answering queries"
    
    def __init__(
        self,
        hybrid_searcher=None,
        reranker=None,
        kg_client=None,
        top_k: int = 10,
        use_reranker: bool = True,
        use_knowledge_graph: bool = True,
        **kwargs,
    ):
        """
        Initialize extractor agent.
        
        Args:
            hybrid_searcher: HybridSearcher instance
            reranker: Reranker instance
            kg_client: KnowledgeGraphClient instance
            top_k: Number of results to retrieve
            use_reranker: Whether to use reranking
            use_knowledge_graph: Whether to use KG enrichment
        """
        super().__init__(**kwargs)
        self._hybrid_searcher = hybrid_searcher
        self._reranker = reranker
        self._kg_client = kg_client
        self.top_k = top_k
        self.use_reranker = use_reranker
        self.use_knowledge_graph = use_knowledge_graph
    
    async def _get_hybrid_searcher(self):
        """Lazy load hybrid searcher."""
        if self._hybrid_searcher is None:
            from src.retrieval.hybrid_search import create_hybrid_searcher
            self._hybrid_searcher = await create_hybrid_searcher()
        return self._hybrid_searcher
    
    async def _get_reranker(self):
        """Lazy load reranker."""
        if self._reranker is None:
            from src.retrieval.reranker import get_reranker
            self._reranker = get_reranker()
        return self._reranker
    
    async def _get_kg_client(self):
        """Lazy load knowledge graph client."""
        if self._kg_client is None:
            from src.pipeline.knowledge_graph import get_knowledge_graph_client
            self._kg_client = get_knowledge_graph_client()
        return self._kg_client
    
    async def process(self, state: AgentState) -> AgentState:
        """
        Extract relevant context for all sub-queries.
        
        Args:
            state: Current agent state
        
        Returns:
            Updated state with retrieved contexts
        """
        sub_queries = state["sub_queries"]
        all_contexts = []
        seen_chunks = set()
        
        # Process each sub-query
        for sq in sub_queries:
            if sq["status"] == StepStatus.COMPLETED.value:
                continue
            
            query = sq["query"]
            sq["status"] = StepStatus.IN_PROGRESS.value
            
            logger.info(f"Extracting context for: {query[:50]}...")
            
            # Hybrid search
            contexts = await self._retrieve_hybrid(query)
            
            # Rerank if enabled
            if self.use_reranker and contexts:
                contexts = await self._rerank_results(query, contexts)
            
            # Knowledge graph enrichment
            kg_context = None
            if self.use_knowledge_graph:
                kg_context = await self._get_kg_context(query, contexts)
            
            # Add to results (deduplicate)
            for ctx in contexts:
                if ctx["chunk_id"] not in seen_chunks:
                    seen_chunks.add(ctx["chunk_id"])
                    all_contexts.append(ctx)
                    sq["sources"].append(ctx["chunk_id"])
        
        # Store results in state
        state["retrieved_contexts"] = all_contexts[:self.top_k * 2]  # Keep top results
        
        if kg_context:
            state["knowledge_graph_context"] = kg_context
        
        # Update sources
        state["sources"] = list(seen_chunks)
        
        # Store agent output
        state["agent_outputs"]["extractor"] = {
            "total_contexts": len(all_contexts),
            "unique_chunks": len(seen_chunks),
            "kg_enriched": kg_context is not None,
        }
        
        logger.info(f"Retrieved {len(all_contexts)} contexts from {len(seen_chunks)} unique chunks")
        
        return state
    
    async def _retrieve_hybrid(self, query: str) -> list[dict]:
        """
        Perform hybrid search.
        
        Args:
            query: Search query
        
        Returns:
            List of retrieved contexts as dicts
        """
        try:
            searcher = await self._get_hybrid_searcher()
            results = await searcher.search(query, top_k=self.top_k)
            
            return [
                {
                    "chunk_id": r.chunk_id,
                    "content": r.content,
                    "score": r.score,
                    "source": ",".join(r.sources),
                    "metadata": r.metadata,
                    "doc_id": r.doc_id,
                    "bm25_score": r.bm25_score,
                    "dense_score": r.dense_score,
                    "sparse_score": r.sparse_score,
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            return []
    
    async def _rerank_results(
        self,
        query: str,
        contexts: list[dict],
    ) -> list[dict]:
        """
        Rerank results using ColBERT.
        
        Args:
            query: Original query
            contexts: Retrieved contexts
        
        Returns:
            Reranked contexts
        """
        try:
            reranker = await self._get_reranker()
            
            # Convert to HybridSearchResult format for reranker
            from src.retrieval.hybrid_search import HybridSearchResult
            
            hybrid_results = [
                HybridSearchResult(
                    doc_id=ctx["doc_id"],
                    chunk_id=ctx["chunk_id"],
                    content=ctx["content"],
                    score=ctx["score"],
                    metadata=ctx.get("metadata", {}),
                )
                for ctx in contexts
            ]
            
            reranked = reranker.rerank(query, hybrid_results, top_k=self.top_k)
            
            return [
                {
                    "chunk_id": r.chunk_id,
                    "content": r.content,
                    "score": r.rerank_score,
                    "source": "reranked",
                    "metadata": r.metadata,
                    "doc_id": r.doc_id,
                    "original_rank": r.original_rank,
                    "new_rank": r.new_rank,
                }
                for r in reranked
            ]
        except Exception as e:
            logger.warning(f"Reranking failed: {e}, using original order")
            return contexts
    
    async def _get_kg_context(
        self,
        query: str,
        contexts: list[dict],
    ) -> dict | None:
        """
        Get knowledge graph context.
        
        Args:
            query: Search query
            contexts: Retrieved contexts
        
        Returns:
            KG context dictionary
        """
        try:
            kg_client = await self._get_kg_client()
            
            # Search for relevant entities
            kg_results = await kg_client.search_entities(query, limit=5)
            
            # Get related entities for top chunks
            related_entities = []
            for ctx in contexts[:3]:
                chunk_context = await kg_client.get_document_context(
                    ctx["chunk_id"],
                    context_hops=2,
                )
                related_entities.extend(chunk_context.get("related", []))
            
            return {
                "entities": [
                    {"name": r.entity.name, "type": r.entity.entity_type, "score": r.score}
                    for r in kg_results
                ],
                "related": related_entities[:10],
            }
        except Exception as e:
            logger.warning(f"KG enrichment failed: {e}")
            return None


__all__ = ["ExtractorAgent"]
