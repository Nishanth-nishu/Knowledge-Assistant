"""
BM25 Sparse Retrieval Module.

Implements BM25 (Best Matching 25) algorithm for lexical/keyword-based retrieval.
This provides exact keyword matching capabilities for the hybrid search system.
"""

import hashlib
import logging
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


@dataclass
class BM25Document:
    """Document representation for BM25 indexing."""
    doc_id: str
    chunk_id: str
    content: str
    tokens: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BM25SearchResult:
    """Single BM25 search result."""
    doc_id: str
    chunk_id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class BM25Tokenizer:
    """
    Text tokenizer for BM25.
    
    Performs:
    - Lowercasing
    - Punctuation removal
    - Whitespace tokenization
    - Optional stopword removal
    - Optional stemming
    """
    
    # Common English stopwords
    STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
        "to", "was", "were", "will", "with", "the", "this", "but", "they",
        "have", "had", "what", "when", "where", "who", "which", "why", "how"
    }
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_stopwords: bool = True,
        min_token_length: int = 2,
    ):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.min_token_length = min_token_length
        
        # Regex for punctuation removal
        self._punct_pattern = re.compile(r'[^\w\s]')
    
    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into a list of tokens."""
        if not text:
            return []
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation
        if self.remove_punctuation:
            text = self._punct_pattern.sub(' ', text)
        
        # Split on whitespace
        tokens = text.split()
        
        # Filter tokens
        filtered = []
        for token in tokens:
            # Skip short tokens
            if len(token) < self.min_token_length:
                continue
            
            # Skip stopwords
            if self.remove_stopwords and token in self.STOPWORDS:
                continue
            
            filtered.append(token)
        
        return filtered
    
    def tokenize_batch(self, texts: list[str]) -> list[list[str]]:
        """Tokenize multiple texts."""
        return [self.tokenize(text) for text in texts]


class BM25Retriever:
    """
    BM25-based sparse retrieval.
    
    Uses the Okapi BM25 algorithm for lexical matching.
    Supports:
    - Document indexing with chunking
    - Query search with top-k results
    - Index persistence (save/load)
    - Incremental updates
    
    Example:
        retriever = BM25Retriever()
        retriever.add_documents([
            BM25Document(doc_id="1", chunk_id="1-0", content="Hello world"),
            BM25Document(doc_id="2", chunk_id="2-0", content="Python programming"),
        ])
        results = retriever.search("python code", top_k=5)
    """
    
    def __init__(
        self,
        tokenizer: BM25Tokenizer | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """
        Initialize BM25 retriever.
        
        Args:
            tokenizer: Custom tokenizer, or None to use default
            k1: BM25 k1 parameter (term frequency saturation)
            b: BM25 b parameter (document length normalization)
        """
        self.tokenizer = tokenizer or BM25Tokenizer()
        self.k1 = k1
        self.b = b
        
        # Index state
        self._documents: list[BM25Document] = []
        self._bm25: BM25Okapi | None = None
        self._is_indexed = False
    
    @property
    def document_count(self) -> int:
        """Return number of indexed documents."""
        return len(self._documents)
    
    @property
    def is_indexed(self) -> bool:
        """Return whether the index is built."""
        return self._is_indexed
    
    def add_documents(
        self,
        documents: list[BM25Document],
        rebuild_index: bool = True,
    ) -> None:
        """
        Add documents to the index.
        
        Args:
            documents: List of documents to add
            rebuild_index: Whether to rebuild BM25 index after adding
        """
        for doc in documents:
            # Tokenize if not already done
            if not doc.tokens:
                doc.tokens = self.tokenizer.tokenize(doc.content)
            self._documents.append(doc)
        
        if rebuild_index:
            self._build_index()
        else:
            self._is_indexed = False
        
        logger.info(f"Added {len(documents)} documents, total: {self.document_count}")
    
    def add_texts(
        self,
        texts: list[str],
        doc_ids: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Convenience method to add raw texts.
        
        Args:
            texts: List of text contents
            doc_ids: Optional document IDs
            metadatas: Optional metadata dicts
        """
        documents = []
        for i, text in enumerate(texts):
            doc_id = doc_ids[i] if doc_ids else str(i)
            metadata = metadatas[i] if metadatas else {}
            
            doc = BM25Document(
                doc_id=doc_id,
                chunk_id=f"{doc_id}-0",
                content=text,
                metadata=metadata,
            )
            documents.append(doc)
        
        self.add_documents(documents)
    
    def _build_index(self) -> None:
        """Build the BM25 index from documents."""
        if not self._documents:
            logger.warning("No documents to index")
            return
        
        # Get tokenized corpus
        corpus = [doc.tokens for doc in self._documents]
        
        # Build BM25 index
        self._bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)
        self._is_indexed = True
        
        logger.info(f"Built BM25 index with {len(corpus)} documents")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float | None = None,
    ) -> list[BM25SearchResult]:
        """
        Search for documents matching the query.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            score_threshold: Minimum score threshold (optional)
        
        Returns:
            List of search results sorted by score (descending)
        """
        if not self._is_indexed or self._bm25 is None:
            raise RuntimeError("Index not built. Call add_documents first.")
        
        # Tokenize query
        query_tokens = self.tokenizer.tokenize(query)
        
        if not query_tokens:
            logger.warning(f"Query tokenized to empty: '{query}'")
            return []
        
        # Get BM25 scores
        scores = self._bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            
            # Apply threshold
            if score_threshold is not None and score < score_threshold:
                continue
            
            # Skip zero scores
            if score <= 0:
                continue
            
            doc = self._documents[idx]
            results.append(BM25SearchResult(
                doc_id=doc.doc_id,
                chunk_id=doc.chunk_id,
                content=doc.content,
                score=score,
                metadata=doc.metadata,
            ))
        
        return results
    
    def search_batch(
        self,
        queries: list[str],
        top_k: int = 10,
    ) -> list[list[BM25SearchResult]]:
        """Search multiple queries."""
        return [self.search(q, top_k=top_k) for q in queries]
    
    def clear(self) -> None:
        """Clear all documents and reset index."""
        self._documents = []
        self._bm25 = None
        self._is_indexed = False
        logger.info("Cleared BM25 index")
    
    def save(self, path: str | Path) -> None:
        """
        Save index to disk.
        
        Args:
            path: Path to save the index
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "documents": self._documents,
            "k1": self.k1,
            "b": self.b,
            "is_indexed": self._is_indexed,
        }
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved BM25 index to {path}")
    
    def load(self, path: str | Path) -> None:
        """
        Load index from disk.
        
        Args:
            path: Path to load the index from
        """
        path = Path(path)
        
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        self._documents = state["documents"]
        self.k1 = state["k1"]
        self.b = state["b"]
        
        if state["is_indexed"]:
            self._build_index()
        
        logger.info(f"Loaded BM25 index from {path} with {self.document_count} documents")
    
    def get_document_by_id(self, chunk_id: str) -> BM25Document | None:
        """Get a document by its chunk ID."""
        for doc in self._documents:
            if doc.chunk_id == chunk_id:
                return doc
        return None


# Convenience function
def create_bm25_retriever(
    k1: float = 1.5,
    b: float = 0.75,
    remove_stopwords: bool = True,
) -> BM25Retriever:
    """Create a configured BM25 retriever."""
    tokenizer = BM25Tokenizer(remove_stopwords=remove_stopwords)
    return BM25Retriever(tokenizer=tokenizer, k1=k1, b=b)


__all__ = [
    "BM25Document",
    "BM25SearchResult",
    "BM25Tokenizer",
    "BM25Retriever",
    "create_bm25_retriever",
]
