"""
Pipeline Package - Document Processing and Indexing.

This package provides:
- Document parsing (PDF, DOCX, HTML, Markdown, Text)
- Intelligent chunking (recursive, sentence, semantic, hierarchical)
- Knowledge graph integration (Neo4j)
- Full indexing pipeline orchestration
"""

from src.pipeline.document_parser import (
    DocumentType,
    ParsedDocument,
    ParsedSection,
    PDFParser,
    DOCXParser,
    HTMLParser,
    MarkdownParser,
    TextParser,
    UnifiedDocumentParser,
    get_document_parser,
)
from src.pipeline.chunking import (
    Chunk,
    ChunkingStrategy,
    RecursiveCharacterChunker,
    SentenceChunker,
    SemanticChunker,
    HierarchicalChunker,
    SlidingWindowChunker,
    DocumentChunker,
    create_chunker,
)
from src.pipeline.knowledge_graph import (
    EntityType,
    RelationType,
    Entity,
    Relationship,
    GraphSearchResult,
    KnowledgeGraphClient,
    EntityExtractor,
    get_knowledge_graph_client,
)
from src.pipeline.indexing import (
    IndexingResult,
    IndexingConfig,
    DocumentIndexer,
    create_document_indexer,
)

__all__ = [
    # Document Parser
    "DocumentType",
    "ParsedDocument",
    "ParsedSection",
    "PDFParser",
    "DOCXParser",
    "HTMLParser",
    "MarkdownParser",
    "TextParser",
    "UnifiedDocumentParser",
    "get_document_parser",
    # Chunking
    "Chunk",
    "ChunkingStrategy",
    "RecursiveCharacterChunker",
    "SentenceChunker",
    "SemanticChunker",
    "HierarchicalChunker",
    "SlidingWindowChunker",
    "DocumentChunker",
    "create_chunker",
    # Knowledge Graph
    "EntityType",
    "RelationType",
    "Entity",
    "Relationship",
    "GraphSearchResult",
    "KnowledgeGraphClient",
    "EntityExtractor",
    "get_knowledge_graph_client",
    # Indexing
    "IndexingResult",
    "IndexingConfig",
    "DocumentIndexer",
    "create_document_indexer",
]
