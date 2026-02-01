"""
Document Indexing Pipeline.

Orchestrates the full document processing workflow:
1. Parse documents (PDF, DOCX, HTML, etc.)
2. Chunk into retrievable pieces
3. Generate embeddings (dense + sparse)
4. Index in vector stores (Qdrant)
5. Build knowledge graph (Neo4j)
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO

from src.config import get_settings
from src.models.embeddings import get_hybrid_embedding_model
from src.pipeline.chunking import Chunk, DocumentChunker, create_chunker
from src.pipeline.document_parser import (
    ParsedDocument,
    UnifiedDocumentParser,
    get_document_parser,
)
from src.pipeline.knowledge_graph import (
    Entity,
    EntityExtractor,
    EntityType,
    KnowledgeGraphClient,
    Relationship,
    RelationType,
    get_knowledge_graph_client,
)
from src.retrieval.dense_retriever import DenseDocument, DenseRetriever
from src.retrieval.bm25_retriever import BM25Document, BM25Retriever
from src.retrieval.sparse_retriever import SparseDocument, SparseRetriever

logger = logging.getLogger(__name__)


@dataclass
class IndexingResult:
    """Result of document indexing."""
    doc_id: str
    filename: str
    success: bool
    chunks_created: int = 0
    entities_extracted: int = 0
    error: str | None = None
    processing_time_ms: float = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexingConfig:
    """Configuration for document indexing."""
    # Chunking
    chunking_strategy: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Embedding
    generate_dense_embeddings: bool = True
    generate_sparse_embeddings: bool = True
    
    # Knowledge Graph
    extract_entities: bool = True
    build_kg_relationships: bool = True
    
    # Storage
    store_in_vector_db: bool = True
    store_in_bm25: bool = True
    store_in_kg: bool = True


class DocumentIndexer:
    """
    Document indexing pipeline.
    
    Orchestrates parsing, chunking, embedding, and storage
    across vector databases and knowledge graph.
    
    Example:
        indexer = DocumentIndexer()
        await indexer.initialize()
        
        result = await indexer.index_file("contracts/agreement.pdf")
        print(f"Indexed {result.chunks_created} chunks")
    """
    
    def __init__(
        self,
        config: IndexingConfig | None = None,
        parser: UnifiedDocumentParser | None = None,
        chunker: DocumentChunker | None = None,
        dense_retriever: DenseRetriever | None = None,
        sparse_retriever: SparseRetriever | None = None,
        bm25_retriever: BM25Retriever | None = None,
        kg_client: KnowledgeGraphClient | None = None,
    ):
        """
        Initialize document indexer.
        
        Args:
            config: Indexing configuration
            parser: Document parser instance
            chunker: Document chunker instance
            dense_retriever: Dense vector retriever
            sparse_retriever: Sparse vector retriever
            bm25_retriever: BM25 retriever
            kg_client: Knowledge graph client
        """
        self.config = config or IndexingConfig()
        
        self._parser = parser
        self._chunker = chunker
        self._dense_retriever = dense_retriever
        self._sparse_retriever = sparse_retriever
        self._bm25_retriever = bm25_retriever
        self._kg_client = kg_client
        self._entity_extractor = None
        self._embedding_model = None
        
        self._initialized = False
    
    @property
    def parser(self) -> UnifiedDocumentParser:
        if self._parser is None:
            self._parser = get_document_parser()
        return self._parser
    
    @property
    def chunker(self) -> DocumentChunker:
        if self._chunker is None:
            self._chunker = create_chunker(
                strategy=self.config.chunking_strategy,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
        return self._chunker
    
    @property
    def entity_extractor(self) -> EntityExtractor:
        if self._entity_extractor is None:
            self._entity_extractor = EntityExtractor()
        return self._entity_extractor
    
    async def initialize(
        self,
        dense_retriever: DenseRetriever | None = None,
        sparse_retriever: SparseRetriever | None = None,
        bm25_retriever: BM25Retriever | None = None,
        kg_client: KnowledgeGraphClient | None = None,
    ) -> None:
        """
        Initialize all components.
        
        Args:
            dense_retriever: Pre-initialized dense retriever
            sparse_retriever: Pre-initialized sparse retriever
            bm25_retriever: Pre-initialized BM25 retriever
            kg_client: Pre-initialized KG client
        """
        if dense_retriever:
            self._dense_retriever = dense_retriever
        if sparse_retriever:
            self._sparse_retriever = sparse_retriever
        if bm25_retriever:
            self._bm25_retriever = bm25_retriever
        if kg_client:
            self._kg_client = kg_client
        
        # Initialize KG if needed
        if self.config.store_in_kg and self._kg_client:
            await self._kg_client.initialize()
        
        self._initialized = True
        logger.info("Document indexer initialized")
    
    async def index_file(
        self,
        file_path: str | Path,
        doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IndexingResult:
        """
        Index a document from file path.
        
        Args:
            file_path: Path to the document
            doc_id: Optional document ID
            metadata: Additional metadata
        
        Returns:
            IndexingResult with processing details
        """
        start_time = datetime.utcnow()
        file_path = Path(file_path)
        
        if not file_path.exists():
            return IndexingResult(
                doc_id=doc_id or str(file_path),
                filename=file_path.name,
                success=False,
                error=f"File not found: {file_path}",
            )
        
        try:
            # Parse document
            parsed = self.parser.parse(file_path, doc_id=doc_id)
            
            # Add extra metadata
            if metadata:
                parsed.metadata.update(metadata)
            
            return await self._index_parsed_document(parsed, start_time)
            
        except Exception as e:
            logger.error(f"Error indexing {file_path}: {e}")
            return IndexingResult(
                doc_id=doc_id or str(file_path),
                filename=file_path.name,
                success=False,
                error=str(e),
            )
    
    async def index_content(
        self,
        content: bytes | str,
        filename: str,
        doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IndexingResult:
        """
        Index document from content bytes.
        
        Args:
            content: Document content
            filename: Original filename
            doc_id: Optional document ID
            metadata: Additional metadata
        
        Returns:
            IndexingResult with processing details
        """
        start_time = datetime.utcnow()
        doc_id = doc_id or str(uuid.uuid4())
        
        try:
            # Parse content
            parsed = self.parser.parse_content(
                content=content,
                filename=filename,
                doc_id=doc_id,
            )
            
            if metadata:
                parsed.metadata.update(metadata)
            
            return await self._index_parsed_document(parsed, start_time)
            
        except Exception as e:
            logger.error(f"Error indexing content: {e}")
            return IndexingResult(
                doc_id=doc_id,
                filename=filename,
                success=False,
                error=str(e),
            )
    
    async def _index_parsed_document(
        self,
        parsed: ParsedDocument,
        start_time: datetime,
    ) -> IndexingResult:
        """Process and index a parsed document."""
        # Chunk document
        chunks = self.chunker.chunk(parsed)
        
        if not chunks:
            return IndexingResult(
                doc_id=parsed.doc_id,
                filename=parsed.filename,
                success=False,
                error="No chunks created from document",
            )
        
        logger.info(f"Created {len(chunks)} chunks for {parsed.doc_id}")
        
        # Run indexing tasks in parallel
        tasks = []
        
        if self.config.store_in_vector_db:
            tasks.append(self._store_in_vector_db(chunks))
        
        if self.config.store_in_bm25 and self._bm25_retriever:
            tasks.append(self._store_in_bm25(chunks))
        
        entities_count = 0
        if self.config.extract_entities and self.config.store_in_kg:
            tasks.append(self._process_knowledge_graph(parsed, chunks))
        
        # Execute all storage tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check for errors
        errors = [str(r) for r in results if isinstance(r, Exception)]
        
        # Get entity count if KG was processed
        for r in results:
            if isinstance(r, int):
                entities_count = r
        
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        return IndexingResult(
            doc_id=parsed.doc_id,
            filename=parsed.filename,
            success=len(errors) == 0,
            chunks_created=len(chunks),
            entities_extracted=entities_count,
            error="; ".join(errors) if errors else None,
            processing_time_ms=processing_time,
            metadata={
                "doc_type": parsed.doc_type.value,
                "word_count": parsed.word_count,
                "page_count": parsed.page_count,
            },
        )
    
    async def _store_in_vector_db(self, chunks: list[Chunk]) -> None:
        """Store chunks in vector databases."""
        # Prepare dense documents
        if self._dense_retriever:
            dense_docs = [
                DenseDocument(
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    metadata=chunk.metadata,
                )
                for chunk in chunks
            ]
            await self._dense_retriever.add_documents(dense_docs)
        
        # Prepare sparse documents
        if self._sparse_retriever:
            sparse_docs = [
                SparseDocument(
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    metadata=chunk.metadata,
                )
                for chunk in chunks
            ]
            await self._sparse_retriever.add_documents(sparse_docs)
    
    async def _store_in_bm25(self, chunks: list[Chunk]) -> None:
        """Store chunks in BM25 index."""
        bm25_docs = [
            BM25Document(
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                metadata=chunk.metadata,
            )
            for chunk in chunks
        ]
        self._bm25_retriever.add_documents(bm25_docs)
    
    async def _process_knowledge_graph(
        self,
        parsed: ParsedDocument,
        chunks: list[Chunk],
    ) -> int:
        """Extract entities and build knowledge graph."""
        if not self._kg_client:
            return 0
        
        all_entities = []
        all_relationships = []
        
        # Create document entity
        doc_entity = Entity(
            entity_id=parsed.doc_id,
            entity_type=EntityType.DOCUMENT,
            name=parsed.filename,
            properties={
                "doc_type": parsed.doc_type.value,
                "word_count": parsed.word_count,
                "page_count": parsed.page_count,
                **parsed.metadata,
            },
        )
        all_entities.append(doc_entity)
        
        # Create chunk entities and extract named entities
        for chunk in chunks:
            # Chunk entity
            chunk_entity = Entity(
                entity_id=chunk.chunk_id,
                entity_type=EntityType.CHUNK,
                name=f"Chunk {chunk.chunk_index + 1}",
                properties={
                    "content": chunk.content[:500],  # Store preview
                    "chunk_index": chunk.chunk_index,
                    "word_count": chunk.word_count,
                },
            )
            all_entities.append(chunk_entity)
            
            # Document -> Chunk relationship
            all_relationships.append(Relationship(
                source_id=parsed.doc_id,
                target_id=chunk.chunk_id,
                relation_type=RelationType.CONTAINS,
            ))
            
            # Extract named entities from chunk
            if self.config.extract_entities:
                extracted = self.entity_extractor.extract(
                    chunk.content, 
                    doc_id=chunk.chunk_id,
                )
                
                for entity in extracted:
                    all_entities.append(entity)
                    
                    # Chunk -> Entity relationship
                    all_relationships.append(Relationship(
                        source_id=chunk.chunk_id,
                        target_id=entity.entity_id,
                        relation_type=RelationType.MENTIONS,
                    ))
        
        # Store in knowledge graph
        await self._kg_client.upsert_entities(all_entities)
        await self._kg_client.create_relationships(all_relationships)
        
        return len(all_entities)
    
    async def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from all indexes.
        
        Args:
            doc_id: Document ID to delete
        
        Returns:
            True if successful
        """
        tasks = []
        
        if self._dense_retriever:
            tasks.append(self._dense_retriever.delete_by_doc_id(doc_id))
        
        if self._sparse_retriever:
            tasks.append(self._sparse_retriever.delete_by_doc_id(doc_id))
        
        if self._kg_client:
            tasks.append(self._kg_client.delete_document(doc_id))
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"Deleted document {doc_id} from all indexes")
        return True
    
    async def index_batch(
        self,
        file_paths: list[str | Path],
        max_concurrent: int = 5,
    ) -> list[IndexingResult]:
        """
        Index multiple documents in parallel.
        
        Args:
            file_paths: List of file paths
            max_concurrent: Max concurrent indexing tasks
        
        Returns:
            List of indexing results
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def index_with_semaphore(path):
            async with semaphore:
                return await self.index_file(path)
        
        tasks = [index_with_semaphore(path) for path in file_paths]
        return await asyncio.gather(*tasks)


# Factory function
async def create_document_indexer(
    config: IndexingConfig | None = None,
    initialize_retrievers: bool = True,
) -> DocumentIndexer:
    """
    Create and initialize a document indexer.
    
    Args:
        config: Indexing configuration
        initialize_retrievers: Whether to initialize retrievers
    
    Returns:
        Configured DocumentIndexer instance
    """
    from src.retrieval.dense_retriever import create_dense_retriever
    from src.retrieval.sparse_retriever import create_sparse_retriever
    from src.retrieval.bm25_retriever import create_bm25_retriever
    
    indexer = DocumentIndexer(config=config)
    
    if initialize_retrievers:
        dense = await create_dense_retriever()
        sparse = await create_sparse_retriever()
        bm25 = create_bm25_retriever()
        kg = get_knowledge_graph_client()
        
        await indexer.initialize(
            dense_retriever=dense,
            sparse_retriever=sparse,
            bm25_retriever=bm25,
            kg_client=kg,
        )
    
    return indexer


__all__ = [
    "IndexingResult",
    "IndexingConfig",
    "DocumentIndexer",
    "create_document_indexer",
]
