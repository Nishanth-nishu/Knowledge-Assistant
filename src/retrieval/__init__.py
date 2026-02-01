"""
Retrieval Package - Hybrid Search and Reranking.

This package implements a three-way hybrid retrieval pipeline:
- BM25 lexical/keyword matching
- Dense vector search (BGE-M3 embeddings)
- SPLADE sparse vector search (learned sparse representations)

Results are fused using Reciprocal Rank Fusion (RRF) and optionally
reranked using ColBERT for improved precision.
"""

from src.retrieval.bm25_retriever import (
    BM25Document,
    BM25Retriever,
    BM25SearchResult,
    BM25Tokenizer,
    create_bm25_retriever,
)
from src.retrieval.dense_retriever import (
    DenseDocument,
    DenseRetriever,
    DenseSearchResult,
    create_dense_retriever,
)
from src.retrieval.sparse_retriever import (
    SparseDocument,
    SparseRetriever,
    SparseSearchResult,
    create_sparse_retriever,
)
from src.retrieval.hybrid_search import (
    HybridSearcher,
    HybridSearcherBuilder,
    HybridSearchResult,
    create_hybrid_searcher,
    reciprocal_rank_fusion,
)
from src.retrieval.reranker import (
    ColBERTReranker,
    CrossEncoderReranker,
    RerankedResult,
    get_cross_encoder_reranker,
    get_reranker,
)

__all__ = [
    # BM25
    "BM25Document",
    "BM25Retriever",
    "BM25SearchResult",
    "BM25Tokenizer",
    "create_bm25_retriever",
    # Dense
    "DenseDocument",
    "DenseRetriever",
    "DenseSearchResult",
    "create_dense_retriever",
    # Sparse
    "SparseDocument",
    "SparseRetriever",
    "SparseSearchResult",
    "create_sparse_retriever",
    # Hybrid
    "HybridSearcher",
    "HybridSearcherBuilder",
    "HybridSearchResult",
    "create_hybrid_searcher",
    "reciprocal_rank_fusion",
    # Reranker
    "ColBERTReranker",
    "CrossEncoderReranker",
    "RerankedResult",
    "get_reranker",
    "get_cross_encoder_reranker",
]
