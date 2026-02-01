"""
Tests for the document processing pipeline.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.pipeline.document_parser import (
    DocumentType,
    ParsedDocument,
    UnifiedDocumentParser,
    MarkdownParser,
    TextParser,
)
from src.pipeline.chunking import (
    Chunk,
    RecursiveCharacterChunker,
    SentenceChunker,
    DocumentChunker,
    create_chunker,
)
from src.pipeline.knowledge_graph import (
    Entity,
    EntityType,
    Relationship,
    RelationType,
)


class TestDocumentParser:
    """Tests for document parsing."""
    
    def test_detect_type(self):
        """Test document type detection."""
        parser = UnifiedDocumentParser()
        
        assert parser.detect_type("doc.pdf") == DocumentType.PDF
        assert parser.detect_type("doc.docx") == DocumentType.DOCX
        assert parser.detect_type("page.html") == DocumentType.HTML
        assert parser.detect_type("readme.md") == DocumentType.MARKDOWN
        assert parser.detect_type("file.txt") == DocumentType.TEXT
        assert parser.detect_type("unknown.xyz") == DocumentType.UNKNOWN
    
    def test_text_parser(self):
        """Test plain text parsing."""
        parser = TextParser()
        
        result = parser.parse(
            file_content="Hello world\n\nThis is a test.",
            doc_id="test-doc",
        )
        
        assert result.doc_id == "test-doc"
        assert result.doc_type == DocumentType.TEXT
        assert "Hello world" in result.content
        assert result.word_count > 0
    
    def test_markdown_parser(self):
        """Test markdown parsing."""
        parser = MarkdownParser()
        
        md_content = """# Title

This is a paragraph.

## Section 1

Some content here.

```python
def hello():
    print("Hello")
```
"""
        
        result = parser.parse(file_content=md_content, doc_id="test-md")
        
        assert result.doc_type == DocumentType.MARKDOWN
        assert len(result.sections) > 0
        
        # Check headings were found
        headings = [s for s in result.sections if s.section_type == "heading"]
        assert len(headings) >= 2
        
        # Check code blocks were found
        code_blocks = [s for s in result.sections if s.section_type == "code"]
        assert len(code_blocks) >= 1


class TestChunking:
    """Tests for document chunking."""
    
    def create_test_document(self, content: str) -> ParsedDocument:
        """Helper to create test document."""
        return ParsedDocument(
            doc_id="test-doc",
            filename="test.txt",
            doc_type=DocumentType.TEXT,
            content=content,
        )
    
    def test_recursive_chunker_basic(self):
        """Test recursive character chunking."""
        chunker = RecursiveCharacterChunker(
            chunk_size=100,
            chunk_overlap=20,
        )
        
        content = "This is sentence one. " * 20
        doc = self.create_test_document(content)
        
        chunks = chunker.chunk(doc)
        
        assert len(chunks) > 1
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.doc_id == "test-doc" for c in chunks)
    
    def test_sentence_chunker(self):
        """Test sentence-based chunking."""
        chunker = SentenceChunker(
            chunk_size=200,
            chunk_overlap=1,
        )
        
        content = "First sentence here. Second sentence follows. Third one is longer and more detailed. Fourth and final."
        doc = self.create_test_document(content)
        
        chunks = chunker.chunk(doc)
        
        assert len(chunks) >= 1
        # Each chunk should end with sentence boundary
        for chunk in chunks:
            assert chunk.content.strip()
    
    def test_document_chunker_factory(self):
        """Test document chunker factory."""
        chunker = create_chunker(
            strategy="recursive",
            chunk_size=500,
            chunk_overlap=50,
        )
        
        assert isinstance(chunker, DocumentChunker)
    
    def test_chunk_structure(self):
        """Test chunk dataclass structure."""
        chunk = Chunk(
            chunk_id="doc-chunk-0",
            doc_id="doc",
            content="Test content here",
            start_char=0,
            end_char=17,
            chunk_index=0,
            total_chunks=5,
        )
        
        assert chunk.char_count == 17
        assert chunk.word_count == 3
        assert chunk.parent_id is None
    
    def test_hierarchical_chunking(self):
        """Test hierarchical parent-child chunking."""
        chunker = create_chunker(
            strategy="hierarchical",
            chunk_size=500,
        )
        
        content = "A" * 1000 + " " + "B" * 1000  # Long content
        doc = self.create_test_document(content)
        
        chunks = chunker.chunk(doc)
        
        # Should have parent and child chunks
        parents = [c for c in chunks if c.metadata.get("is_parent")]
        children = [c for c in chunks if not c.metadata.get("is_parent")]
        
        assert len(parents) > 0
        assert len(children) > 0


class TestKnowledgeGraphEntities:
    """Tests for knowledge graph entities."""
    
    def test_entity_creation(self):
        """Test entity creation."""
        entity = Entity(
            entity_id="person-1",
            entity_type=EntityType.PERSON,
            name="John Doe",
            properties={"role": "CEO"},
        )
        
        assert entity.entity_id == "person-1"
        assert entity.entity_type == EntityType.PERSON
        assert entity.name == "John Doe"
    
    def test_entity_to_dict(self):
        """Test entity serialization."""
        entity = Entity(
            entity_id="org-1",
            entity_type=EntityType.ORGANIZATION,
            name="Acme Corp",
            properties={"industry": "Tech"},
        )
        
        data = entity.to_dict()
        
        assert data["entity_id"] == "org-1"
        assert data["entity_type"] == "Organization"
        assert data["name"] == "Acme Corp"
        assert data["industry"] == "Tech"
    
    def test_relationship_creation(self):
        """Test relationship creation."""
        rel = Relationship(
            source_id="doc-1",
            target_id="chunk-1",
            relation_type=RelationType.CONTAINS,
            confidence=0.95,
        )
        
        assert rel.source_id == "doc-1"
        assert rel.target_id == "chunk-1"
        assert rel.confidence == 0.95
    
    def test_relationship_to_dict(self):
        """Test relationship serialization."""
        rel = Relationship(
            source_id="chunk-1",
            target_id="entity-1",
            relation_type=RelationType.MENTIONS,
            properties={"count": 3},
        )
        
        data = rel.to_dict()
        
        assert data["relation_type"] == "MENTIONS"
        assert data["count"] == 3


class TestIndexingPipeline:
    """Tests for the indexing pipeline."""
    
    def test_indexing_config_defaults(self):
        """Test indexing config default values."""
        from src.pipeline.indexing import IndexingConfig
        
        config = IndexingConfig()
        
        assert config.chunking_strategy == "recursive"
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.generate_dense_embeddings is True
        assert config.extract_entities is True
    
    def test_indexing_result_structure(self):
        """Test indexing result structure."""
        from src.pipeline.indexing import IndexingResult
        
        result = IndexingResult(
            doc_id="doc-1",
            filename="test.pdf",
            success=True,
            chunks_created=10,
            entities_extracted=25,
            processing_time_ms=1500.5,
        )
        
        assert result.success is True
        assert result.chunks_created == 10
        assert result.error is None
