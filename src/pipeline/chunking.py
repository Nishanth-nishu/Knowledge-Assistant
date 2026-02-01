"""
Intelligent Document Chunking Module.

Provides multiple chunking strategies for RAG:
- Recursive character splitting
- Semantic chunking (using embeddings)
- Sentence-based chunking
- Sliding window chunking
- Hierarchical/parent-child chunking

Optimized for retrieval quality and context preservation.
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable
from uuid import uuid4

from src.pipeline.document_parser import ParsedDocument, ParsedSection

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A document chunk ready for indexing."""
    chunk_id: str
    doc_id: str
    content: str
    start_char: int = 0
    end_char: int = 0
    chunk_index: int = 0
    total_chunks: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Hierarchical relationships
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    
    @property
    def char_count(self) -> int:
        return len(self.content)
    
    @property
    def word_count(self) -> int:
        return len(self.content.split())


class ChunkingStrategy(ABC):
    """Base class for chunking strategies."""
    
    @abstractmethod
    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        """Chunk a parsed document into smaller pieces."""
        pass
    
    def _generate_chunk_id(self, doc_id: str, index: int) -> str:
        """Generate a unique chunk ID."""
        return f"{doc_id}-chunk-{index}"


class RecursiveCharacterChunker(ChunkingStrategy):
    """
    Recursive character text splitter.
    
    Tries to split by paragraphs, then sentences, then words,
    ensuring chunks don't exceed max size while preserving context.
    """
    
    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
        length_function: Callable[[str], int] | None = None,
    ):
        """
        Initialize recursive chunker.
        
        Args:
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks
            separators: List of separators to try, in order
            length_function: Custom function to measure text length
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
        self.length_function = length_function or len
    
    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        """Chunk document using recursive splitting."""
        text = document.content
        chunks_text = self._split_text(text, self.separators)
        
        chunks = []
        char_pos = 0
        
        for i, chunk_text in enumerate(chunks_text):
            # Find position in original text
            start = text.find(chunk_text, char_pos)
            if start == -1:
                start = char_pos
            end = start + len(chunk_text)
            char_pos = end - self.chunk_overlap
            
            chunks.append(Chunk(
                chunk_id=self._generate_chunk_id(document.doc_id, i),
                doc_id=document.doc_id,
                content=chunk_text,
                start_char=start,
                end_char=end,
                chunk_index=i,
                total_chunks=len(chunks_text),
                metadata={
                    "filename": document.filename,
                    "doc_type": document.doc_type.value,
                    **document.metadata,
                },
            ))
        
        # Update total_chunks now that we know it
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        logger.debug(f"Created {len(chunks)} chunks from {document.doc_id}")
        return chunks
    
    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text."""
        final_chunks: list[str] = []
        
        # Find the appropriate separator
        separator = separators[-1]  # Default to last
        for sep in separators:
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                break
        
        # Split by separator
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)
        
        # Merge or split further
        good_splits: list[str] = []
        for split in splits:
            if self.length_function(split) < self.chunk_size:
                good_splits.append(split)
            else:
                # Need to split further
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                
                # Recursively split
                remaining_separators = separators[separators.index(separator) + 1:]
                if remaining_separators:
                    sub_chunks = self._split_text(split, remaining_separators)
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(split)
        
        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)
        
        return final_chunks
    
    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        """Merge small splits into larger chunks with overlap."""
        chunks = []
        current = []
        current_len = 0
        
        for split in splits:
            split_len = self.length_function(split)
            
            if current_len + split_len + len(separator) > self.chunk_size:
                if current:
                    chunks.append(separator.join(current))
                    
                    # Keep overlap
                    while current and current_len > self.chunk_overlap:
                        removed = current.pop(0)
                        current_len -= self.length_function(removed) + len(separator)
                
                current = [split] if current_len == 0 else current + [split]
                current_len = sum(self.length_function(s) for s in current) + len(separator) * (len(current) - 1)
            else:
                current.append(split)
                current_len += split_len + (len(separator) if current_len > 0 else 0)
        
        if current:
            chunks.append(separator.join(current))
        
        return chunks


class SentenceChunker(ChunkingStrategy):
    """
    Sentence-based chunking.
    
    Groups sentences together until reaching chunk size,
    respecting sentence boundaries.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 1,  # Number of sentences to overlap
        min_sentences: int = 1,
        max_sentences: int = 20,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        
        # Sentence splitting pattern
        self._sentence_pattern = re.compile(
            r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$'
        )
    
    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        sentences = self._sentence_pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        """Chunk by sentences."""
        sentences = self._split_sentences(document.content)
        
        chunks = []
        current_sentences = []
        current_len = 0
        char_pos = 0
        
        for sent in sentences:
            sent_len = len(sent)
            
            # Check if adding this sentence exceeds limits
            if (current_len + sent_len > self.chunk_size and 
                len(current_sentences) >= self.min_sentences) or \
               len(current_sentences) >= self.max_sentences:
                # Create chunk
                chunk_text = " ".join(current_sentences)
                start = document.content.find(current_sentences[0], char_pos)
                
                chunks.append(Chunk(
                    chunk_id=self._generate_chunk_id(document.doc_id, len(chunks)),
                    doc_id=document.doc_id,
                    content=chunk_text,
                    start_char=start,
                    end_char=start + len(chunk_text),
                    chunk_index=len(chunks),
                    metadata={
                        "filename": document.filename,
                        "sentence_count": len(current_sentences),
                    },
                ))
                
                # Overlap
                overlap_sentences = current_sentences[-self.chunk_overlap:] if self.chunk_overlap > 0 else []
                current_sentences = overlap_sentences
                current_len = sum(len(s) for s in current_sentences)
                char_pos = start
            
            current_sentences.append(sent)
            current_len += sent_len
        
        # Final chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            start = document.content.find(current_sentences[0], char_pos)
            
            chunks.append(Chunk(
                chunk_id=self._generate_chunk_id(document.doc_id, len(chunks)),
                doc_id=document.doc_id,
                content=chunk_text,
                start_char=start,
                end_char=start + len(chunk_text),
                chunk_index=len(chunks),
                metadata={
                    "filename": document.filename,
                    "sentence_count": len(current_sentences),
                },
            ))
        
        # Update total_chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks


class SemanticChunker(ChunkingStrategy):
    """
    Semantic chunking using embeddings.
    
    Groups text by semantic similarity, creating chunks
    that contain coherent ideas/topics.
    """
    
    def __init__(
        self,
        embedding_model=None,
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
    ):
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def _get_embedding_model(self):
        """Lazy load embedding model."""
        if self.embedding_model is None:
            from src.models.embeddings import get_dense_embedding_model
            self.embedding_model = get_dense_embedding_model()
        return self.embedding_model
    
    def _cosine_similarity(self, v1, v2) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
    
    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        """Chunk based on semantic similarity."""
        # Split into sentences first
        sentence_pattern = re.compile(r'(?<=[.!?])\s+')
        sentences = sentence_pattern.split(document.content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return []
        
        # Get embeddings for all sentences
        model = self._get_embedding_model()
        result = model.embed(sentences)
        embeddings = result.embeddings
        
        # Group sentences by semantic similarity
        chunks = []
        current_sentences = [sentences[0]]
        current_embedding = embeddings[0]
        char_pos = 0
        
        for i in range(1, len(sentences)):
            sent = sentences[i]
            emb = embeddings[i]
            
            similarity = self._cosine_similarity(current_embedding, emb)
            current_len = sum(len(s) for s in current_sentences)
            
            # Decide whether to start new chunk
            should_break = (
                similarity < self.similarity_threshold and 
                current_len >= self.min_chunk_size
            ) or current_len + len(sent) > self.max_chunk_size
            
            if should_break:
                # Create chunk
                chunk_text = " ".join(current_sentences)
                start = document.content.find(current_sentences[0], char_pos)
                
                chunks.append(Chunk(
                    chunk_id=self._generate_chunk_id(document.doc_id, len(chunks)),
                    doc_id=document.doc_id,
                    content=chunk_text,
                    start_char=start,
                    end_char=start + len(chunk_text),
                    chunk_index=len(chunks),
                    metadata={
                        "filename": document.filename,
                        "chunking_method": "semantic",
                    },
                ))
                
                current_sentences = [sent]
                current_embedding = emb
                char_pos = start
            else:
                current_sentences.append(sent)
                # Update running average embedding
                import numpy as np
                current_embedding = np.mean([current_embedding, emb], axis=0)
        
        # Final chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            start = document.content.find(current_sentences[0], char_pos)
            
            chunks.append(Chunk(
                chunk_id=self._generate_chunk_id(document.doc_id, len(chunks)),
                doc_id=document.doc_id,
                content=chunk_text,
                start_char=start,
                end_char=start + len(chunk_text),
                chunk_index=len(chunks),
                metadata={
                    "filename": document.filename,
                    "chunking_method": "semantic",
                },
            ))
        
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks


class HierarchicalChunker(ChunkingStrategy):
    """
    Hierarchical chunking with parent-child relationships.
    
    Creates large parent chunks and smaller child chunks,
    useful for retrieval with context expansion.
    """
    
    def __init__(
        self,
        parent_chunk_size: int = 2000,
        child_chunk_size: int = 400,
        child_overlap: int = 50,
    ):
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.child_overlap = child_overlap
        
        self._parent_chunker = RecursiveCharacterChunker(
            chunk_size=parent_chunk_size,
            chunk_overlap=0,
        )
        self._child_chunker = RecursiveCharacterChunker(
            chunk_size=child_chunk_size,
            chunk_overlap=child_overlap,
        )
    
    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        """Create hierarchical chunks."""
        all_chunks = []
        
        # Create parent chunks
        parent_chunks = self._parent_chunker.chunk(document)
        
        for parent in parent_chunks:
            parent.metadata["is_parent"] = True
            all_chunks.append(parent)
            
            # Create child chunks within parent
            child_doc = ParsedDocument(
                doc_id=parent.chunk_id,
                filename=document.filename,
                doc_type=document.doc_type,
                content=parent.content,
                metadata=parent.metadata,
            )
            
            child_chunks = self._child_chunker.chunk(child_doc)
            
            for child in child_chunks:
                child.parent_id = parent.chunk_id
                child.chunk_id = f"{parent.chunk_id}-child-{child.chunk_index}"
                child.doc_id = document.doc_id
                child.metadata["is_parent"] = False
                child.metadata["parent_chunk_id"] = parent.chunk_id
                
                all_chunks.append(child)
            
            parent.children_ids = [c.chunk_id for c in child_chunks]
        
        return all_chunks


class SlidingWindowChunker(ChunkingStrategy):
    """
    Sliding window chunking.
    
    Creates overlapping chunks with fixed size and step.
    Good for dense retrieval where boundary issues matter.
    """
    
    def __init__(
        self,
        window_size: int = 500,
        step_size: int = 250,
    ):
        self.window_size = window_size
        self.step_size = step_size
    
    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        """Create sliding window chunks."""
        text = document.content
        chunks = []
        
        for i in range(0, len(text), self.step_size):
            chunk_text = text[i:i + self.window_size]
            
            if len(chunk_text.strip()) < 50:  # Skip tiny chunks
                continue
            
            chunks.append(Chunk(
                chunk_id=self._generate_chunk_id(document.doc_id, len(chunks)),
                doc_id=document.doc_id,
                content=chunk_text,
                start_char=i,
                end_char=min(i + self.window_size, len(text)),
                chunk_index=len(chunks),
                metadata={
                    "filename": document.filename,
                    "chunking_method": "sliding_window",
                },
            ))
        
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks


class DocumentChunker:
    """
    Main document chunking interface.
    
    Provides easy access to different chunking strategies
    with sensible defaults for RAG applications.
    """
    
    def __init__(
        self,
        strategy: str = "recursive",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **kwargs,
    ):
        """
        Initialize document chunker.
        
        Args:
            strategy: Chunking strategy ("recursive", "sentence", "semantic", 
                     "hierarchical", "sliding_window")
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks
            **kwargs: Additional strategy-specific parameters
        """
        strategies = {
            "recursive": RecursiveCharacterChunker,
            "sentence": SentenceChunker,
            "semantic": SemanticChunker,
            "hierarchical": HierarchicalChunker,
            "sliding_window": SlidingWindowChunker,
        }
        
        if strategy not in strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(strategies.keys())}")
        
        # Build kwargs based on strategy
        if strategy in ("recursive", "sentence"):
            self._chunker = strategies[strategy](
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                **kwargs,
            )
        elif strategy == "hierarchical":
            self._chunker = strategies[strategy](
                parent_chunk_size=kwargs.get("parent_chunk_size", chunk_size * 2),
                child_chunk_size=kwargs.get("child_chunk_size", chunk_size // 2),
                child_overlap=kwargs.get("child_overlap", chunk_overlap),
            )
        elif strategy == "sliding_window":
            self._chunker = strategies[strategy](
                window_size=chunk_size,
                step_size=chunk_size - chunk_overlap,
            )
        else:
            self._chunker = strategies[strategy](**kwargs)
    
    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        """Chunk a document."""
        return self._chunker.chunk(document)
    
    def chunk_text(
        self,
        text: str,
        doc_id: str = "text",
        filename: str = "text.txt",
    ) -> list[Chunk]:
        """Convenience method to chunk raw text."""
        from src.pipeline.document_parser import DocumentType
        
        doc = ParsedDocument(
            doc_id=doc_id,
            filename=filename,
            doc_type=DocumentType.TEXT,
            content=text,
        )
        return self.chunk(doc)


# Factory function
def create_chunker(
    strategy: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> DocumentChunker:
    """Create a document chunker with specified strategy."""
    return DocumentChunker(
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


__all__ = [
    "Chunk",
    "ChunkingStrategy",
    "RecursiveCharacterChunker",
    "SentenceChunker",
    "SemanticChunker",
    "HierarchicalChunker",
    "SlidingWindowChunker",
    "DocumentChunker",
    "create_chunker",
]
