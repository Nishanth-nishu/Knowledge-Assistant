"""
Tests for the retrieval pipeline.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from src.retrieval.bm25_retriever import (
    BM25Document,
    BM25Retriever,
    BM25Tokenizer,
    create_bm25_retriever,
)
from src.retrieval.hybrid_search import reciprocal_rank_fusion


class TestBM25Tokenizer:
    """Tests for BM25Tokenizer."""
    
    def test_basic_tokenization(self):
        """Test basic text tokenization."""
        tokenizer = BM25Tokenizer()
        tokens = tokenizer.tokenize("Hello World, this is a test!")
        
        assert len(tokens) > 0
        assert all(t.islower() for t in tokens)  # All lowercase
        assert "," not in "".join(tokens)  # No punctuation
    
    def test_stopword_removal(self):
        """Test stopword removal."""
        tokenizer = BM25Tokenizer(remove_stopwords=True)
        tokens = tokenizer.tokenize("The quick brown fox is a test")
        
        # Common stopwords should be removed
        assert "the" not in tokens
        assert "is" not in tokens
        assert "a" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens
    
    def test_min_token_length(self):
        """Test minimum token length filtering."""
        tokenizer = BM25Tokenizer(min_token_length=3)
        tokens = tokenizer.tokenize("I am a test of short tokens")
        
        assert all(len(t) >= 3 for t in tokens)
    
    def test_empty_input(self):
        """Test empty string handling."""
        tokenizer = BM25Tokenizer()
        tokens = tokenizer.tokenize("")
        
        assert tokens == []
    
    def test_batch_tokenization(self):
        """Test batch tokenization."""
        tokenizer = BM25Tokenizer()
        texts = ["Hello world", "Python programming"]
        
        batch = tokenizer.tokenize_batch(texts)
        
        assert len(batch) == 2
        assert isinstance(batch[0], list)
        assert isinstance(batch[1], list)


class TestBM25Retriever:
    """Tests for BM25Retriever."""
    
    def test_add_documents(self):
        """Test adding documents."""
        retriever = BM25Retriever()
        docs = [
            BM25Document(doc_id="1", chunk_id="1-0", content="Machine learning algorithms"),
            BM25Document(doc_id="2", chunk_id="2-0", content="Natural language processing"),
        ]
        
        retriever.add_documents(docs)
        
        assert retriever.document_count == 2
        assert retriever.is_indexed
    
    def test_add_texts(self):
        """Test adding raw texts."""
        retriever = BM25Retriever()
        texts = ["Document one", "Document two", "Document three"]
        
        retriever.add_texts(texts)
        
        assert retriever.document_count == 3
    
    def test_search(self):
        """Test basic search."""
        retriever = BM25Retriever()
        retriever.add_texts([
            "Machine learning is a subset of artificial intelligence",
            "Natural language processing deals with text",
            "Computer vision processes images",
        ])
        
        results = retriever.search("machine learning", top_k=2)
        
        assert len(results) <= 2
        assert results[0].score > 0
        # First result should contain "machine" or "learning"
        assert "machine" in results[0].content.lower() or "learning" in results[0].content.lower()
    
    def test_search_empty_index(self):
        """Test search on empty index raises error."""
        retriever = BM25Retriever()
        
        with pytest.raises(RuntimeError):
            retriever.search("test query")
    
    def test_search_with_threshold(self):
        """Test search with score threshold."""
        retriever = BM25Retriever()
        retriever.add_texts([
            "The quick brown fox",
            "The lazy dog sleeps",
        ])
        
        # Very high threshold should return no results
        results = retriever.search("fox", top_k=10, score_threshold=100.0)
        
        assert len(results) == 0
    
    def test_clear(self):
        """Test clearing the index."""
        retriever = BM25Retriever()
        retriever.add_texts(["test document"])
        
        assert retriever.document_count == 1
        
        retriever.clear()
        
        assert retriever.document_count == 0
        assert not retriever.is_indexed
    
    def test_get_document_by_id(self):
        """Test retrieving document by ID."""
        retriever = BM25Retriever()
        docs = [
            BM25Document(doc_id="doc1", chunk_id="doc1-0", content="First doc"),
            BM25Document(doc_id="doc2", chunk_id="doc2-0", content="Second doc"),
        ]
        retriever.add_documents(docs)
        
        found = retriever.get_document_by_id("doc1-0")
        
        assert found is not None
        assert found.content == "First doc"
        
        not_found = retriever.get_document_by_id("nonexistent")
        assert not_found is None


class TestReciprocalRankFusion:
    """Tests for RRF algorithm."""
    
    def test_single_list(self):
        """Test RRF with single result list."""
        results = [
            [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)],
        ]
        
        fused = reciprocal_rank_fusion(results, k=60)
        
        assert len(fused) == 3
        assert fused[0][0] == "doc1"  # First in input should be first in output
    
    def test_multiple_lists(self):
        """Test RRF with multiple lists."""
        results = [
            [("doc1", 0.9), ("doc2", 0.8)],
            [("doc2", 0.95), ("doc3", 0.85)],
        ]
        
        fused = reciprocal_rank_fusion(results, k=60)
        
        # doc2 appears in both lists, should score higher
        assert len(fused) == 3
        assert fused[0][0] == "doc2"
    
    def test_empty_lists(self):
        """Test RRF with empty input."""
        fused = reciprocal_rank_fusion([])
        
        assert fused == []
    
    def test_weighted_fusion(self):
        """Test RRF with custom weights."""
        results = [
            [("doc1", 0.9)],
            [("doc2", 0.95)],
        ]
        weights = [0.3, 0.7]  # Heavily favor second list
        
        fused = reciprocal_rank_fusion(results, k=60, weights=weights)
        
        assert len(fused) == 2
        # doc2 should be higher due to higher weight
        assert fused[0][0] == "doc2"
    
    def test_different_k_values(self):
        """Test RRF with different k values."""
        results = [
            [("doc1", 0.9), ("doc2", 0.8)],
            [("doc2", 0.95), ("doc1", 0.85)],
        ]
        
        # Different k values should produce same ordering but different scores
        fused_k1 = reciprocal_rank_fusion(results, k=1)
        fused_k100 = reciprocal_rank_fusion(results, k=100)
        
        # Same documents in both
        assert set(d for d, _ in fused_k1) == set(d for d, _ in fused_k100)


class TestCreateBM25Retriever:
    """Tests for factory function."""
    
    def test_create_default(self):
        """Test creating retriever with defaults."""
        retriever = create_bm25_retriever()
        
        assert isinstance(retriever, BM25Retriever)
        assert retriever.k1 == 1.5
        assert retriever.b == 0.75
    
    def test_create_custom_params(self):
        """Test creating retriever with custom parameters."""
        retriever = create_bm25_retriever(k1=2.0, b=0.5, remove_stopwords=False)
        
        assert retriever.k1 == 2.0
        assert retriever.b == 0.5


# Mock tests for async retrievers (require infrastructure)
class TestDenseRetrieverMock:
    """Mock tests for DenseRetriever."""
    
    @pytest.mark.asyncio
    async def test_search_mock(self):
        """Test dense search with mocked Qdrant."""
        from src.retrieval.dense_retriever import DenseRetriever
        
        retriever = DenseRetriever()
        
        # Mock the client and embedding model
        mock_client = MagicMock()
        mock_client.search.return_value = [
            MagicMock(
                score=0.95,
                payload={
                    "doc_id": "doc1",
                    "chunk_id": "doc1-0",
                    "content": "Test content",
                }
            )
        ]
        retriever._client = mock_client
        retriever._initialized = True
        
        # Mock embedding model
        mock_embedding = MagicMock()
        mock_embedding.embed_query.return_value = [0.1] * 1024
        retriever._embedding_model = mock_embedding
        
        results = await retriever.search("test query", top_k=5)
        
        assert len(results) == 1
        assert results[0].doc_id == "doc1"
        assert results[0].score == 0.95
