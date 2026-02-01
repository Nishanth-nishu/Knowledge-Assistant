"""
SPLADE Sparse Vector Retrieval.

Implements sparse vector retrieval using SPLADE (SParse Lexical AnD Expansion)
for learned sparse representations that combine lexical and semantic matching.
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from src.config import get_qdrant_settings
from src.models.embeddings import SparseEmbeddingModel, get_sparse_embedding_model

logger = logging.getLogger(__name__)


@dataclass
class SparseDocument:
    """Document for sparse retrieval."""
    doc_id: str
    chunk_id: str
    content: str
    sparse_indices: list[int] | None = None
    sparse_values: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SparseSearchResult:
    """Sparse search result."""
    doc_id: str
    chunk_id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class SparseRetriever:
    """
    SPLADE-based sparse vector retrieval.
    
    Uses learned sparse representations for improved lexical matching
    with semantic expansion. Stored in Qdrant as sparse vectors.
    
    Features:
    - SPLADE sparse embeddings with term expansion
    - Qdrant sparse vector storage and search
    - Batch operations for efficiency
    - Combined with dense vectors for hybrid search
    
    Example:
        retriever = SparseRetriever()
        await retriever.initialize()
        
        await retriever.add_documents([
            SparseDocument(doc_id="1", chunk_id="1-0", content="Machine learning"),
        ])
        
        results = await retriever.search("ML algorithms", top_k=5)
    """
    
    SPARSE_VECTOR_NAME = "sparse"
    
    def __init__(
        self,
        collection_name: str | None = None,
        embedding_model: SparseEmbeddingModel | None = None,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
    ):
        """
        Initialize sparse retriever.
        
        Args:
            collection_name: Qdrant collection name
            embedding_model: SPLADE embedding model
            qdrant_url: Qdrant server URL
            qdrant_api_key: Qdrant API key
        """
        self._settings = get_qdrant_settings()
        
        self.collection_name = collection_name or f"{self._settings.collection_name}_sparse"
        self.qdrant_url = qdrant_url or self._settings.url
        self.qdrant_api_key = qdrant_api_key or (
            self._settings.api_key.get_secret_value() if self._settings.api_key else None
        )
        
        # Lazy loading
        self._client: QdrantClient | None = None
        self._embedding_model = embedding_model
        self._initialized = False
    
    @property
    def client(self) -> QdrantClient:
        """Get or create Qdrant client."""
        if self._client is None:
            self._client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                timeout=self._settings.timeout,
                prefer_grpc=self._settings.prefer_grpc,
            )
        return self._client
    
    @property
    def embedding_model(self) -> SparseEmbeddingModel:
        """Get or create sparse embedding model."""
        if self._embedding_model is None:
            self._embedding_model = get_sparse_embedding_model()
        return self._embedding_model
    
    async def initialize(self, recreate: bool = False) -> None:
        """
        Initialize the collection.
        
        Args:
            recreate: If True, delete and recreate the collection
        """
        if recreate:
            await self.delete_collection()
        
        await self._ensure_collection()
        self._initialized = True
        logger.info(f"Initialized sparse retriever with collection '{self.collection_name}'")
    
    async def _ensure_collection(self) -> None:
        """Ensure collection exists with sparse vector support."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            await self._create_collection()
    
    async def _create_collection(self) -> None:
        """Create collection with sparse vector configuration."""
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={},  # No dense vectors
            sparse_vectors_config={
                self.SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=False,
                    )
                )
            },
        )
        
        # Create payload indexes
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="doc_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        
        logger.info(f"Created sparse collection '{self.collection_name}'")
    
    async def delete_collection(self) -> None:
        """Delete the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}'")
        except UnexpectedResponse:
            logger.debug(f"Collection '{self.collection_name}' does not exist")
    
    async def add_documents(
        self,
        documents: list[SparseDocument],
        batch_size: int = 100,
    ) -> None:
        """
        Add documents with sparse vectors.
        
        Args:
            documents: Documents to add
            batch_size: Batch size for upsert operations
        """
        if not documents:
            return
        
        # Generate sparse embeddings for documents without them
        texts_to_embed = []
        embed_indices = []
        
        for i, doc in enumerate(documents):
            if doc.sparse_indices is None or doc.sparse_values is None:
                texts_to_embed.append(doc.content)
                embed_indices.append(i)
        
        if texts_to_embed:
            result = self.embedding_model.embed(texts_to_embed)
            for i, idx in enumerate(embed_indices):
                documents[idx].sparse_indices = result.indices[i]
                documents[idx].sparse_values = result.values[i]
        
        # Prepare points
        points = []
        for doc in documents:
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"sparse-{doc.chunk_id}"))
            
            payload = {
                "doc_id": doc.doc_id,
                "chunk_id": doc.chunk_id,
                "content": doc.content,
                **doc.metadata,
            }
            
            # Create sparse vector
            sparse_vector = models.SparseVector(
                indices=doc.sparse_indices,
                values=doc.sparse_values,
            )
            
            points.append(models.PointStruct(
                id=point_id,
                payload=payload,
                vector={self.SPARSE_VECTOR_NAME: sparse_vector},
            ))
        
        # Batch upsert
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )
        
        logger.info(f"Added {len(documents)} sparse documents to '{self.collection_name}'")
    
    async def add_texts(
        self,
        texts: list[str],
        doc_ids: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Convenience method to add raw texts."""
        documents = []
        for i, text in enumerate(texts):
            doc_id = doc_ids[i] if doc_ids else str(i)
            metadata = metadatas[i] if metadatas else {}
            
            doc = SparseDocument(
                doc_id=doc_id,
                chunk_id=f"{doc_id}-{i}",
                content=text,
                metadata=metadata,
            )
            documents.append(doc)
        
        await self.add_documents(documents)
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float | None = None,
        filter_conditions: dict[str, Any] | None = None,
    ) -> list[SparseSearchResult]:
        """
        Search using sparse vectors.
        
        Args:
            query: Search query
            top_k: Number of results
            score_threshold: Minimum score threshold
            filter_conditions: Metadata filters
        
        Returns:
            List of search results
        """
        # Generate sparse query embedding
        query_indices, query_values = self.embedding_model.embed_query(query)
        
        # Build filter
        query_filter = None
        if filter_conditions:
            must_conditions = []
            for key, value in filter_conditions.items():
                if isinstance(value, list):
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchAny(any=value),
                        )
                    )
                else:
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value),
                        )
                    )
            query_filter = models.Filter(must=must_conditions)
        
        # Create sparse query vector
        sparse_query = models.SparseVector(
            indices=query_indices,
            values=query_values,
        )
        
        # Search with sparse vector
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=models.NamedSparseVector(
                name=self.SPARSE_VECTOR_NAME,
                vector=sparse_query,
            ),
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=query_filter,
            with_payload=True,
        )
        
        # Convert to result objects
        search_results = []
        for hit in results:
            payload = hit.payload or {}
            search_results.append(SparseSearchResult(
                doc_id=payload.get("doc_id", ""),
                chunk_id=payload.get("chunk_id", ""),
                content=payload.get("content", ""),
                score=hit.score,
                metadata={k: v for k, v in payload.items()
                         if k not in ("doc_id", "chunk_id", "content")},
            ))
        
        return search_results
    
    async def search_by_sparse_vector(
        self,
        indices: list[int],
        values: list[float],
        top_k: int = 10,
    ) -> list[SparseSearchResult]:
        """Search using pre-computed sparse vector."""
        sparse_query = models.SparseVector(indices=indices, values=values)
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=models.NamedSparseVector(
                name=self.SPARSE_VECTOR_NAME,
                vector=sparse_query,
            ),
            limit=top_k,
            with_payload=True,
        )
        
        search_results = []
        for hit in results:
            payload = hit.payload or {}
            search_results.append(SparseSearchResult(
                doc_id=payload.get("doc_id", ""),
                chunk_id=payload.get("chunk_id", ""),
                content=payload.get("content", ""),
                score=hit.score,
                metadata={k: v for k, v in payload.items()
                         if k not in ("doc_id", "chunk_id", "content")},
            ))
        
        return search_results
    
    async def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all sparse vectors for a document."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchValue(value=doc_id),
                        )
                    ]
                )
            ),
        )
        logger.info(f"Deleted document '{doc_id}' from sparse collection")
    
    async def get_collection_info(self) -> dict[str, Any]:
        """Get collection information."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": info.points_count,
            "status": info.status.value,
        }
    
    def close(self) -> None:
        """Close the client connection."""
        if self._client:
            self._client.close()
            self._client = None


# Factory function
async def create_sparse_retriever(
    collection_name: str | None = None,
    recreate: bool = False,
) -> SparseRetriever:
    """Create and initialize a sparse retriever."""
    retriever = SparseRetriever(collection_name=collection_name)
    await retriever.initialize(recreate=recreate)
    return retriever


__all__ = [
    "SparseDocument",
    "SparseSearchResult",
    "SparseRetriever",
    "create_sparse_retriever",
]
