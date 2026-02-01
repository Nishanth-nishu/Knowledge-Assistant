"""
Hybrid Search with Reciprocal Rank Fusion (RRF).

Implements three-way hybrid search combining:
- BM25 lexical search (exact keyword matching)
- Dense vector search (semantic similarity)
- SPLADE sparse vector search (learned sparse representations)

Results are fused using Reciprocal Rank Fusion (RRF) algorithm.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from src.config import get_settings
from src.retrieval.bm25_retriever import BM25Retriever, BM25SearchResult
from src.retrieval.dense_retriever import DenseRetriever, DenseSearchResult
from src.retrieval.sparse_retriever import SparseRetriever, SparseSearchResult

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchResult:
    """Unified search result from hybrid retrieval."""
    doc_id: str
    chunk_id: str
    content: str
    score: float  # RRF fused score
    
    # Individual scores
    bm25_score: float | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    
    # Ranking info
    bm25_rank: int | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def sources(self) -> list[str]:
        """Return which retrievers found this document."""
        sources = []
        if self.bm25_rank is not None:
            sources.append("bm25")
        if self.dense_rank is not None:
            sources.append("dense")
        if self.sparse_rank is not None:
            sources.append("sparse")
        return sources


def reciprocal_rank_fusion(
    result_lists: list[list[tuple[str, float]]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion algorithm.
    
    Combines multiple ranked lists into a single fused ranking.
    
    Formula: RRF(d) = Σ (weight_i / (k + rank_i(d)))
    
    Args:
        result_lists: List of ranked results, each as [(doc_id, score), ...]
        k: RRF constant (default 60, as per original paper)
        weights: Optional weights for each result list
    
    Returns:
        Fused ranked list as [(doc_id, rrf_score), ...]
    """
    if not result_lists:
        return []
    
    # Default to equal weights
    if weights is None:
        weights = [1.0] * len(result_lists)
    
    # Normalize weights
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]
    
    # Calculate RRF scores
    rrf_scores: dict[str, float] = {}
    
    for list_idx, results in enumerate(result_lists):
        weight = weights[list_idx]
        for rank, (doc_id, _) in enumerate(results, start=1):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += weight / (k + rank)
    
    # Sort by RRF score
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_results


class HybridSearcher:
    """
    Three-way hybrid search combining BM25, dense, and sparse retrieval.
    
    Uses Reciprocal Rank Fusion (RRF) to combine results from:
    1. BM25 - Traditional lexical matching
    2. Dense vectors (BGE-M3) - Semantic similarity
    3. SPLADE sparse vectors - Learned sparse matching
    
    Features:
    - Configurable weights for each retriever
    - Parallel execution for performance
    - Result deduplication and merging
    - Score normalization
    
    Example:
        searcher = HybridSearcher(
            bm25_retriever=bm25,
            dense_retriever=dense,
            sparse_retriever=sparse,
        )
        
        results = await searcher.search("legal contract terms", top_k=10)
    """
    
    def __init__(
        self,
        bm25_retriever: BM25Retriever | None = None,
        dense_retriever: DenseRetriever | None = None,
        sparse_retriever: SparseRetriever | None = None,
        bm25_weight: float | None = None,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
        rrf_k: int | None = None,
    ):
        """
        Initialize hybrid searcher.
        
        Args:
            bm25_retriever: BM25 retriever instance
            dense_retriever: Dense vector retriever instance
            sparse_retriever: Sparse vector retriever instance
            bm25_weight: Weight for BM25 results (0-1)
            dense_weight: Weight for dense results (0-1)
            sparse_weight: Weight for sparse results (0-1)
            rrf_k: RRF constant (higher = more uniform fusion)
        """
        self._settings = get_settings().retrieval
        
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        
        # Weights with defaults from config
        self.bm25_weight = bm25_weight if bm25_weight is not None else self._settings.bm25_weight
        self.dense_weight = dense_weight if dense_weight is not None else self._settings.dense_weight
        self.sparse_weight = sparse_weight if sparse_weight is not None else self._settings.sparse_weight
        self.rrf_k = rrf_k if rrf_k is not None else self._settings.rrf_k
        
        # Cache for document content
        self._doc_cache: dict[str, dict[str, Any]] = {}
    
    @property
    def weights(self) -> list[float]:
        """Return active retriever weights."""
        weights = []
        if self.bm25_retriever:
            weights.append(self.bm25_weight)
        if self.dense_retriever:
            weights.append(self.dense_weight)
        if self.sparse_retriever:
            weights.append(self.sparse_weight)
        return weights
    
    async def search(
        self,
        query: str,
        top_k: int = 20,
        use_bm25: bool = True,
        use_dense: bool = True,
        use_sparse: bool = True,
        filter_conditions: dict[str, Any] | None = None,
    ) -> list[HybridSearchResult]:
        """
        Perform hybrid search.
        
        Args:
            query: Search query
            top_k: Number of final results to return
            use_bm25: Include BM25 results
            use_dense: Include dense vector results
            use_sparse: Include sparse vector results
            filter_conditions: Metadata filters (for dense/sparse)
        
        Returns:
            List of hybrid search results fused with RRF
        """
        # Fetch more results than needed for fusion
        fetch_k = top_k * 3
        
        # Run searches in parallel
        tasks = []
        retriever_names = []
        
        if use_bm25 and self.bm25_retriever:
            tasks.append(self._search_bm25(query, fetch_k))
            retriever_names.append("bm25")
        
        if use_dense and self.dense_retriever:
            tasks.append(self._search_dense(query, fetch_k, filter_conditions))
            retriever_names.append("dense")
        
        if use_sparse and self.sparse_retriever:
            tasks.append(self._search_sparse(query, fetch_k, filter_conditions))
            retriever_names.append("sparse")
        
        if not tasks:
            logger.warning("No retrievers available for search")
            return []
        
        # Execute searches in parallel
        search_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        result_lists: list[list[tuple[str, float]]] = []
        weights: list[float] = []
        
        all_docs: dict[str, dict[str, Any]] = {}
        rank_info: dict[str, dict[str, Any]] = {}
        
        for i, (name, results) in enumerate(zip(retriever_names, search_results)):
            if isinstance(results, Exception):
                logger.error(f"Search error in {name}: {results}")
                continue
            
            result_list: list[tuple[str, float]] = []
            
            for rank, result in enumerate(results, start=1):
                chunk_id = result.chunk_id
                
                # Store document info
                if chunk_id not in all_docs:
                    all_docs[chunk_id] = {
                        "doc_id": result.doc_id,
                        "chunk_id": chunk_id,
                        "content": result.content,
                        "metadata": result.metadata,
                    }
                
                # Store rank info
                if chunk_id not in rank_info:
                    rank_info[chunk_id] = {}
                rank_info[chunk_id][f"{name}_rank"] = rank
                rank_info[chunk_id][f"{name}_score"] = result.score
                
                result_list.append((chunk_id, result.score))
            
            result_lists.append(result_list)
            
            # Assign weight based on retriever type
            if name == "bm25":
                weights.append(self.bm25_weight)
            elif name == "dense":
                weights.append(self.dense_weight)
            else:
                weights.append(self.sparse_weight)
        
        # Apply RRF fusion
        fused = reciprocal_rank_fusion(result_lists, k=self.rrf_k, weights=weights)
        
        # Build final results
        results = []
        for chunk_id, rrf_score in fused[:top_k]:
            doc_info = all_docs[chunk_id]
            ranks = rank_info.get(chunk_id, {})
            
            results.append(HybridSearchResult(
                doc_id=doc_info["doc_id"],
                chunk_id=chunk_id,
                content=doc_info["content"],
                score=rrf_score,
                bm25_score=ranks.get("bm25_score"),
                dense_score=ranks.get("dense_score"),
                sparse_score=ranks.get("sparse_score"),
                bm25_rank=ranks.get("bm25_rank"),
                dense_rank=ranks.get("dense_rank"),
                sparse_rank=ranks.get("sparse_rank"),
                metadata=doc_info["metadata"],
            ))
        
        logger.debug(f"Hybrid search returned {len(results)} results for query: {query[:50]}...")
        return results
    
    async def _search_bm25(
        self,
        query: str,
        top_k: int,
    ) -> list[BM25SearchResult]:
        """Execute BM25 search."""
        return self.bm25_retriever.search(query, top_k=top_k)
    
    async def _search_dense(
        self,
        query: str,
        top_k: int,
        filter_conditions: dict[str, Any] | None,
    ) -> list[DenseSearchResult]:
        """Execute dense vector search."""
        return await self.dense_retriever.search(
            query, top_k=top_k, filter_conditions=filter_conditions
        )
    
    async def _search_sparse(
        self,
        query: str,
        top_k: int,
        filter_conditions: dict[str, Any] | None,
    ) -> list[SparseSearchResult]:
        """Execute sparse vector search."""
        return await self.sparse_retriever.search(
            query, top_k=top_k, filter_conditions=filter_conditions
        )
    
    async def search_with_scores(
        self,
        query: str,
        top_k: int = 20,
    ) -> tuple[list[HybridSearchResult], dict[str, Any]]:
        """
        Search with detailed scoring information.
        
        Returns results plus a summary of retrieval statistics.
        """
        results = await self.search(query, top_k=top_k)
        
        # Gather statistics
        stats = {
            "query": query,
            "total_results": len(results),
            "bm25_only": sum(1 for r in results if r.sources == ["bm25"]),
            "dense_only": sum(1 for r in results if r.sources == ["dense"]),
            "sparse_only": sum(1 for r in results if r.sources == ["sparse"]),
            "multi_source": sum(1 for r in results if len(r.sources) > 1),
            "avg_rrf_score": sum(r.score for r in results) / len(results) if results else 0,
        }
        
        return results, stats


class HybridSearcherBuilder:
    """Builder for creating HybridSearcher with proper initialization."""
    
    def __init__(self):
        self._bm25: BM25Retriever | None = None
        self._dense: DenseRetriever | None = None
        self._sparse: SparseRetriever | None = None
        self._bm25_weight: float | None = None
        self._dense_weight: float | None = None
        self._sparse_weight: float | None = None
        self._rrf_k: int | None = None
    
    def with_bm25(
        self,
        retriever: BM25Retriever | None = None,
        weight: float | None = None,
    ) -> "HybridSearcherBuilder":
        """Add BM25 retriever."""
        from src.retrieval.bm25_retriever import create_bm25_retriever
        self._bm25 = retriever or create_bm25_retriever()
        self._bm25_weight = weight
        return self
    
    async def with_dense(
        self,
        retriever: DenseRetriever | None = None,
        collection_name: str | None = None,
        weight: float | None = None,
    ) -> "HybridSearcherBuilder":
        """Add dense vector retriever."""
        if retriever:
            self._dense = retriever
        else:
            from src.retrieval.dense_retriever import create_dense_retriever
            self._dense = await create_dense_retriever(collection_name=collection_name)
        self._dense_weight = weight
        return self
    
    async def with_sparse(
        self,
        retriever: SparseRetriever | None = None,
        collection_name: str | None = None,
        weight: float | None = None,
    ) -> "HybridSearcherBuilder":
        """Add sparse vector retriever."""
        if retriever:
            self._sparse = retriever
        else:
            from src.retrieval.sparse_retriever import create_sparse_retriever
            self._sparse = await create_sparse_retriever(collection_name=collection_name)
        self._sparse_weight = weight
        return self
    
    def with_rrf_k(self, k: int) -> "HybridSearcherBuilder":
        """Set RRF k parameter."""
        self._rrf_k = k
        return self
    
    def build(self) -> HybridSearcher:
        """Build the hybrid searcher."""
        return HybridSearcher(
            bm25_retriever=self._bm25,
            dense_retriever=self._dense,
            sparse_retriever=self._sparse,
            bm25_weight=self._bm25_weight,
            dense_weight=self._dense_weight,
            sparse_weight=self._sparse_weight,
            rrf_k=self._rrf_k,
        )


# Factory functions
async def create_hybrid_searcher(
    use_bm25: bool = True,
    use_dense: bool = True,
    use_sparse: bool = True,
) -> HybridSearcher:
    """
    Create a fully configured hybrid searcher.
    
    Args:
        use_bm25: Include BM25 retriever
        use_dense: Include dense vector retriever
        use_sparse: Include sparse vector retriever
    
    Returns:
        Configured HybridSearcher instance
    """
    builder = HybridSearcherBuilder()
    
    if use_bm25:
        builder.with_bm25()
    
    if use_dense:
        await builder.with_dense()
    
    if use_sparse:
        await builder.with_sparse()
    
    return builder.build()


__all__ = [
    "HybridSearchResult",
    "HybridSearcher",
    "HybridSearcherBuilder",
    "reciprocal_rank_fusion",
    "create_hybrid_searcher",
]
