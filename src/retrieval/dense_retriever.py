"""
Dense Vector Retrieval with Qdrant.

Implements dense semantic retrieval using Qdrant vector database.
Uses BGE-M3 embeddings for semantic similarity search.
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from src.config import get_qdrant_settings
from src.models.embeddings import DenseEmbeddingModel, get_dense_embedding_model

logger = logging.getLogger(__name__)


@dataclass
class DenseDocument:
    """Document for dense retrieval."""
    doc_id: str
    chunk_id: str
    content: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DenseSearchResult:
    """Dense search result."""
    doc_id: str
    chunk_id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class DenseRetriever:
    """
    Dense vector retrieval using Qdrant.
    
    Features:
    - Semantic similarity search using dense embeddings
    - Collection management (create, delete, info)
    - Batch upsert and search operations
    - Metadata filtering support
    - Scroll/pagination for large result sets
    
    Example:
        retriever = DenseRetriever()
        await retriever.initialize()
        
        await retriever.add_documents([
            DenseDocument(doc_id="1", chunk_id="1-0", content="Hello world"),
        ])
        
        results = await retriever.search("greeting", top_k=5)
    """
    
    def __init__(
        self,
        collection_name: str | None = None,
        embedding_model: DenseEmbeddingModel | None = None,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
        vector_size: int | None = None,
        distance: str | None = None,
    ):
        """
        Initialize dense retriever.
        
        Args:
            collection_name: Qdrant collection name
            embedding_model: Embedding model instance
            qdrant_url: Qdrant server URL
            qdrant_api_key: Qdrant API key
            vector_size: Vector dimension
            distance: Distance metric (Cosine, Euclid, Dot)
        """
        self._settings = get_qdrant_settings()
        
        self.collection_name = collection_name or self._settings.collection_name
        self.qdrant_url = qdrant_url or self._settings.url
        self.qdrant_api_key = qdrant_api_key or (
            self._settings.api_key.get_secret_value() if self._settings.api_key else None
        )
        self.vector_size = vector_size or self._settings.vector_size
        self.distance = distance or self._settings.distance
        
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
    def embedding_model(self) -> DenseEmbeddingModel:
        """Get or create embedding model."""
        if self._embedding_model is None:
            self._embedding_model = get_dense_embedding_model()
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
        logger.info(f"Initialized dense retriever with collection '{self.collection_name}'")
    
    async def _ensure_collection(self) -> None:
        """Ensure collection exists."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            await self._create_collection()
    
    async def _create_collection(self) -> None:
        """Create the Qdrant collection."""
        distance_map = {
            "Cosine": models.Distance.COSINE,
            "Euclid": models.Distance.EUCLID,
            "Dot": models.Distance.DOT,
        }
        
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=distance_map.get(self.distance, models.Distance.COSINE),
            ),
            # Optimized index config for production
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=100,
                full_scan_threshold=10000,
            ),
            # Payload indexing for filtering
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=20000,
            ),
        )
        
        # Create payload indexes for common filter fields
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="doc_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        
        logger.info(f"Created collection '{self.collection_name}' with {self.vector_size}d vectors")
    
    async def delete_collection(self) -> None:
        """Delete the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}'")
        except UnexpectedResponse:
            logger.debug(f"Collection '{self.collection_name}' does not exist")
    
    async def add_documents(
        self,
        documents: list[DenseDocument],
        batch_size: int = 100,
    ) -> None:
        """
        Add documents to the collection.
        
        Args:
            documents: Documents to add
            batch_size: Batch size for upsert operations
        """
        if not documents:
            return
        
        # Generate embeddings for documents without them
        texts_to_embed = []
        embed_indices = []
        
        for i, doc in enumerate(documents):
            if doc.embedding is None:
                texts_to_embed.append(doc.content)
                embed_indices.append(i)
        
        if texts_to_embed:
            result = self.embedding_model.embed(texts_to_embed, is_query=False)
            for idx, embedding in zip(embed_indices, result.embeddings):
                documents[idx].embedding = embedding.tolist()
        
        # Prepare points
        points = []
        for doc in documents:
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.chunk_id))
            
            payload = {
                "doc_id": doc.doc_id,
                "chunk_id": doc.chunk_id,
                "content": doc.content,
                **doc.metadata,
            }
            
            points.append(models.PointStruct(
                id=point_id,
                vector=doc.embedding,
                payload=payload,
            ))
        
        # Batch upsert
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )
        
        logger.info(f"Added {len(documents)} documents to '{self.collection_name}'")
    
    async def add_texts(
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
            
            doc = DenseDocument(
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
    ) -> list[DenseSearchResult]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of results
            score_threshold: Minimum similarity score
            filter_conditions: Metadata filter conditions
        
        Returns:
            List of search results
        """
        # Embed query
        query_embedding = self.embedding_model.embed_query(query)
        
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
        
        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=query_filter,
            with_payload=True,
        )
        
        # Convert to result objects
        search_results = []
        for hit in results:
            payload = hit.payload or {}
            search_results.append(DenseSearchResult(
                doc_id=payload.get("doc_id", ""),
                chunk_id=payload.get("chunk_id", ""),
                content=payload.get("content", ""),
                score=hit.score,
                metadata={k: v for k, v in payload.items() 
                         if k not in ("doc_id", "chunk_id", "content")},
            ))
        
        return search_results
    
    async def search_by_vector(
        self,
        vector: list[float],
        top_k: int = 10,
        score_threshold: float | None = None,
    ) -> list[DenseSearchResult]:
        """Search using a pre-computed vector."""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        
        search_results = []
        for hit in results:
            payload = hit.payload or {}
            search_results.append(DenseSearchResult(
                doc_id=payload.get("doc_id", ""),
                chunk_id=payload.get("chunk_id", ""),
                content=payload.get("content", ""),
                score=hit.score,
                metadata={k: v for k, v in payload.items()
                         if k not in ("doc_id", "chunk_id", "content")},
            ))
        
        return search_results
    
    async def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all chunks for a document."""
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
        logger.info(f"Deleted document '{doc_id}' from '{self.collection_name}'")
    
    async def get_collection_info(self) -> dict[str, Any]:
        """Get collection information."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": info.status.value,
        }
    
    def close(self) -> None:
        """Close the client connection."""
        if self._client:
            self._client.close()
            self._client = None


# Factory function
async def create_dense_retriever(
    collection_name: str | None = None,
    recreate: bool = False,
) -> DenseRetriever:
    """Create and initialize a dense retriever."""
    retriever = DenseRetriever(collection_name=collection_name)
    await retriever.initialize(recreate=recreate)
    return retriever


__all__ = [
    "DenseDocument",
    "DenseSearchResult",
    "DenseRetriever",
    "create_dense_retriever",
]
