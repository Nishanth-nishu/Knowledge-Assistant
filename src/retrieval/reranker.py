"""
ColBERT Reranker for Retrieved Documents.

Implements cross-encoder reranking using ColBERT-style models
for improved precision after initial hybrid retrieval.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from src.config import get_settings
from src.retrieval.hybrid_search import HybridSearchResult

logger = logging.getLogger(__name__)


@dataclass
class RerankedResult:
    """Reranked search result."""
    doc_id: str
    chunk_id: str
    content: str
    rerank_score: float
    original_score: float
    original_rank: int
    new_rank: int
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def rank_change(self) -> int:
        """How much the ranking changed (positive = moved up)."""
        return self.original_rank - self.new_rank


class ColBERTReranker:
    """
    ColBERT-style reranker for search results.
    
    Uses late interaction between query and document token embeddings
    for more accurate relevance scoring than bi-encoder approaches.
    
    Features:
    - Late interaction scoring (MaxSim)
    - Batch processing for efficiency
    - GPU acceleration when available
    - Configurable top-k selection
    
    Example:
        reranker = ColBERTReranker()
        reranked = reranker.rerank(query, hybrid_results, top_k=10)
    """
    
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 16,
    ):
        """
        Initialize ColBERT reranker.
        
        Args:
            model_name: Model name/path (default from config)
            device: Device to use (cuda/cpu)
            max_length: Maximum sequence length
            batch_size: Batch size for processing
        """
        self._settings = get_settings().reranker
        
        self.model_name = model_name or self._settings.model
        self.device = device or self._settings.device
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Lazy loading
        self._model = None
        self._tokenizer = None
        
        logger.info(f"Initializing ColBERTReranker: {self.model_name} on {self.device}")
    
    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        if self._model is not None:
            return
        
        logger.info(f"Loading reranker model: {self.model_name}")
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()
        
        logger.info(f"Reranker model loaded on {self.device}")
    
    @property
    def model(self):
        """Get model with lazy loading."""
        if self._model is None:
            self._load_model()
        return self._model
    
    @property
    def tokenizer(self):
        """Get tokenizer with lazy loading."""
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer
    
    def _encode(self, texts: list[str]) -> torch.Tensor:
        """Encode texts to token embeddings."""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use last hidden state
            embeddings = outputs.last_hidden_state
            
            # Apply attention mask
            mask = inputs["attention_mask"].unsqueeze(-1)
            embeddings = embeddings * mask
        
        return embeddings
    
    def _maxsim_score(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
    ) -> float:
        """
        Calculate MaxSim score (ColBERT late interaction).
        
        For each query token, find the maximum similarity with any document token,
        then sum across all query tokens.
        
        Args:
            query_embeddings: [1, query_len, dim]
            doc_embeddings: [1, doc_len, dim]
        
        Returns:
            MaxSim score
        """
        # Remove batch dimension
        q_emb = query_embeddings.squeeze(0)  # [query_len, dim]
        d_emb = doc_embeddings.squeeze(0)    # [doc_len, dim]
        
        # Normalize embeddings
        q_emb = torch.nn.functional.normalize(q_emb, p=2, dim=-1)
        d_emb = torch.nn.functional.normalize(d_emb, p=2, dim=-1)
        
        # Compute similarity matrix: [query_len, doc_len]
        sim_matrix = torch.matmul(q_emb, d_emb.transpose(0, 1))
        
        # MaxSim: max over document tokens for each query token
        max_sim = sim_matrix.max(dim=1).values  # [query_len]
        
        # Sum over query tokens
        score = max_sim.sum().item()
        
        return score
    
    def _score_pair(self, query: str, document: str) -> float:
        """Score a single query-document pair."""
        query_emb = self._encode([query])
        doc_emb = self._encode([document])
        return self._maxsim_score(query_emb, doc_emb)
    
    def _score_batch(
        self,
        query: str,
        documents: list[str],
    ) -> list[float]:
        """Score multiple documents against a single query."""
        # Encode query once
        query_emb = self._encode([query])
        
        scores = []
        for doc in documents:
            doc_emb = self._encode([doc])
            score = self._maxsim_score(query_emb, doc_emb)
            scores.append(score)
        
        return scores
    
    def rerank(
        self,
        query: str,
        results: list[HybridSearchResult],
        top_k: int | None = None,
    ) -> list[RerankedResult]:
        """
        Rerank search results using ColBERT.
        
        Args:
            query: Search query
            results: Initial search results from hybrid retrieval
            top_k: Number of top results to return (None = all)
        
        Returns:
            Reranked results sorted by rerank score
        """
        if not results:
            return []
        
        top_k = top_k or self._settings.top_k
        
        # Extract documents
        documents = [r.content for r in results]
        
        # Score all documents
        logger.debug(f"Reranking {len(documents)} documents")
        scores = self._score_batch(query, documents)
        
        # Create reranked results
        reranked = []
        for i, (result, score) in enumerate(zip(results, scores)):
            reranked.append(RerankedResult(
                doc_id=result.doc_id,
                chunk_id=result.chunk_id,
                content=result.content,
                rerank_score=score,
                original_score=result.score,
                original_rank=i + 1,
                new_rank=0,  # Will be set after sorting
                metadata=result.metadata,
            ))
        
        # Sort by rerank score
        reranked.sort(key=lambda x: x.rerank_score, reverse=True)
        
        # Assign new ranks
        for i, r in enumerate(reranked):
            r.new_rank = i + 1
        
        # Return top-k
        return reranked[:top_k]
    
    def rerank_simple(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """
        Simple reranking without HybridSearchResult objects.
        
        Args:
            query: Search query
            documents: List of document texts
            top_k: Number of top results
        
        Returns:
            List of (original_index, score) tuples sorted by score
        """
        if not documents:
            return []
        
        top_k = top_k or len(documents)
        
        scores = self._score_batch(query, documents)
        
        # Create (index, score) pairs and sort
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        return indexed_scores[:top_k]


class CrossEncoderReranker:
    """
    Alternative reranker using cross-encoder models.
    
    Uses models like ms-marco-MiniLM or bge-reranker for
    direct relevance scoring.
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
        max_length: int = 512,
    ):
        """Initialize cross-encoder reranker."""
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        
        # Lazy loading
        self._model = None
        self._tokenizer = None
    
    def _load_model(self) -> None:
        """Load the cross-encoder model."""
        if self._model is not None:
            return
        
        from transformers import AutoModelForSequenceClassification
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()
    
    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model
    
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer
    
    def score(self, query: str, documents: list[str]) -> list[float]:
        """Score query-document pairs."""
        pairs = [[query, doc] for doc in documents]
        
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = outputs.logits.squeeze(-1).cpu().tolist()
        
        # Handle single result case
        if isinstance(scores, float):
            scores = [scores]
        
        return scores
    
    def rerank(
        self,
        query: str,
        results: list[HybridSearchResult],
        top_k: int | None = None,
    ) -> list[RerankedResult]:
        """Rerank using cross-encoder."""
        if not results:
            return []
        
        top_k = top_k or len(results)
        
        documents = [r.content for r in results]
        scores = self.score(query, documents)
        
        reranked = []
        for i, (result, score) in enumerate(zip(results, scores)):
            reranked.append(RerankedResult(
                doc_id=result.doc_id,
                chunk_id=result.chunk_id,
                content=result.content,
                rerank_score=score,
                original_score=result.score,
                original_rank=i + 1,
                new_rank=0,
                metadata=result.metadata,
            ))
        
        reranked.sort(key=lambda x: x.rerank_score, reverse=True)
        
        for i, r in enumerate(reranked):
            r.new_rank = i + 1
        
        return reranked[:top_k]


# Singleton instance
_reranker: ColBERTReranker | None = None


def get_reranker() -> ColBERTReranker:
    """Get singleton reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = ColBERTReranker()
    return _reranker


def get_cross_encoder_reranker() -> CrossEncoderReranker:
    """Get a cross-encoder reranker."""
    return CrossEncoderReranker()


__all__ = [
    "RerankedResult",
    "ColBERTReranker",
    "CrossEncoderReranker",
    "get_reranker",
    "get_cross_encoder_reranker",
]
