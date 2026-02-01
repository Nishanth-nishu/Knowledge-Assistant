"""
Neo4j Knowledge Graph Integration.

Provides entity and relationship management for building
a knowledge graph from documents. Enables:
- Entity extraction and storage
- Relationship mapping
- Graph-based retrieval
- Dual-channel search (Vector + Knowledge Graph)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from neo4j import AsyncGraphDatabase, GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from src.config import get_neo4j_settings

logger = logging.getLogger(__name__)


class EntityType(str, Enum):
    """Common entity types for knowledge graphs."""
    PERSON = "Person"
    ORGANIZATION = "Organization"
    LOCATION = "Location"
    DATE = "Date"
    DOCUMENT = "Document"
    CHUNK = "Chunk"
    CONCEPT = "Concept"
    EVENT = "Event"
    PRODUCT = "Product"
    REGULATION = "Regulation"
    POLICY = "Policy"
    CLAUSE = "Clause"


class RelationType(str, Enum):
    """Common relationship types."""
    MENTIONS = "MENTIONS"
    CONTAINS = "CONTAINS"
    RELATED_TO = "RELATED_TO"
    PART_OF = "PART_OF"
    AUTHORED_BY = "AUTHORED_BY"
    REFERENCES = "REFERENCES"
    SUPERSEDES = "SUPERSEDES"
    PARENT_OF = "PARENT_OF"
    SIMILAR_TO = "SIMILAR_TO"
    DEFINES = "DEFINES"
    APPLIES_TO = "APPLIES_TO"


@dataclass
class Entity:
    """Knowledge graph entity."""
    entity_id: str
    entity_type: EntityType | str
    name: str
    properties: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for Neo4j."""
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type.value if isinstance(self.entity_type, EntityType) else self.entity_type,
            "name": self.name,
            **self.properties,
        }


@dataclass
class Relationship:
    """Knowledge graph relationship."""
    source_id: str
    target_id: str
    relation_type: RelationType | str
    properties: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "relation_type": self.relation_type.value if isinstance(self.relation_type, RelationType) else self.relation_type,
            "confidence": self.confidence,
            **self.properties,
        }


@dataclass
class GraphSearchResult:
    """Result from knowledge graph search."""
    entity: Entity
    score: float
    path: list[str] = field(default_factory=list)
    related_entities: list[Entity] = field(default_factory=list)
    context: str = ""


class KnowledgeGraphClient:
    """
    Neo4j Knowledge Graph client.
    
    Provides CRUD operations for entities and relationships,
    as well as graph traversal and search capabilities.
    
    Example:
        client = KnowledgeGraphClient()
        await client.initialize()
        
        # Add entity
        entity = Entity(
            entity_id="doc1",
            entity_type=EntityType.DOCUMENT,
            name="Contract Agreement",
        )
        await client.upsert_entity(entity)
        
        # Search related entities
        results = await client.find_related("doc1", max_depth=2)
    """
    
    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ):
        """
        Initialize knowledge graph client.
        
        Args:
            uri: Neo4j URI
            user: Username
            password: Password
            database: Database name
        """
        self._settings = get_neo4j_settings()
        
        self.uri = uri or self._settings.uri
        self.user = user or self._settings.user
        self.password = password or (
            self._settings.password.get_secret_value() if self._settings.password else ""
        )
        self.database = database or self._settings.database
        
        self._driver = None
        self._async_driver = None
    
    @property
    def driver(self):
        """Get synchronous driver."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
        return self._driver
    
    @property
    def async_driver(self):
        """Get async driver."""
        if self._async_driver is None:
            self._async_driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
        return self._async_driver
    
    async def initialize(self) -> None:
        """Initialize database with indexes and constraints."""
        async with self.async_driver.session(database=self.database) as session:
            # Create constraints for entity uniqueness
            constraints = [
                "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
                "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.entity_id IS UNIQUE",
                "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.entity_id IS UNIQUE",
            ]
            
            for constraint in constraints:
                try:
                    await session.run(constraint)
                except Exception as e:
                    logger.debug(f"Constraint may already exist: {e}")
            
            # Create indexes
            indexes = [
                "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
                "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
                "CREATE FULLTEXT INDEX entity_search IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.description]",
            ]
            
            for index in indexes:
                try:
                    await session.run(index)
                except Exception as e:
                    logger.debug(f"Index may already exist: {e}")
        
        logger.info("Initialized Neo4j knowledge graph")
    
    async def upsert_entity(self, entity: Entity) -> None:
        """
        Create or update an entity.
        
        Args:
            entity: Entity to upsert
        """
        entity_type = entity.entity_type.value if isinstance(entity.entity_type, EntityType) else entity.entity_type
        
        query = f"""
        MERGE (e:{entity_type} {{entity_id: $entity_id}})
        SET e += $properties
        SET e.name = $name
        SET e.entity_type = $entity_type
        SET e.updated_at = datetime()
        """
        
        async with self.async_driver.session(database=self.database) as session:
            await session.run(
                query,
                entity_id=entity.entity_id,
                name=entity.name,
                entity_type=entity_type,
                properties=entity.properties,
            )
    
    async def upsert_entities(self, entities: list[Entity]) -> None:
        """Batch upsert entities."""
        for entity in entities:
            await self.upsert_entity(entity)
    
    async def create_relationship(self, relationship: Relationship) -> None:
        """
        Create a relationship between entities.
        
        Args:
            relationship: Relationship to create
        """
        rel_type = relationship.relation_type.value if isinstance(relationship.relation_type, RelationType) else relationship.relation_type
        
        query = f"""
        MATCH (source {{entity_id: $source_id}})
        MATCH (target {{entity_id: $target_id}})
        MERGE (source)-[r:{rel_type}]->(target)
        SET r += $properties
        SET r.confidence = $confidence
        SET r.created_at = datetime()
        """
        
        async with self.async_driver.session(database=self.database) as session:
            await session.run(
                query,
                source_id=relationship.source_id,
                target_id=relationship.target_id,
                properties=relationship.properties,
                confidence=relationship.confidence,
            )
    
    async def create_relationships(self, relationships: list[Relationship]) -> None:
        """Batch create relationships."""
        for rel in relationships:
            await self.create_relationship(rel)
    
    async def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID."""
        query = """
        MATCH (e {entity_id: $entity_id})
        RETURN e
        """
        
        async with self.async_driver.session(database=self.database) as session:
            result = await session.run(query, entity_id=entity_id)
            record = await result.single()
            
            if record:
                node = record["e"]
                return Entity(
                    entity_id=node["entity_id"],
                    entity_type=node.get("entity_type", "Entity"),
                    name=node.get("name", ""),
                    properties=dict(node),
                )
        
        return None
    
    async def find_related(
        self,
        entity_id: str,
        relation_types: list[str] | None = None,
        max_depth: int = 2,
        limit: int = 50,
    ) -> list[Entity]:
        """
        Find entities related to a given entity.
        
        Args:
            entity_id: Source entity ID
            relation_types: Filter by relationship types
            max_depth: Maximum traversal depth
            limit: Maximum results
        
        Returns:
            List of related entities
        """
        rel_pattern = ""
        if relation_types:
            rel_types = "|".join(relation_types)
            rel_pattern = f"[:{rel_types}*1..{max_depth}]"
        else:
            rel_pattern = f"[*1..{max_depth}]"
        
        query = f"""
        MATCH (source {{entity_id: $entity_id}})-{rel_pattern}-(related)
        WHERE source <> related
        RETURN DISTINCT related
        LIMIT $limit
        """
        
        entities = []
        async with self.async_driver.session(database=self.database) as session:
            result = await session.run(
                query,
                entity_id=entity_id,
                limit=limit,
            )
            
            async for record in result:
                node = record["related"]
                entities.append(Entity(
                    entity_id=node.get("entity_id", ""),
                    entity_type=node.get("entity_type", "Entity"),
                    name=node.get("name", ""),
                    properties=dict(node),
                ))
        
        return entities
    
    async def search_entities(
        self,
        query: str,
        entity_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[GraphSearchResult]:
        """
        Full-text search for entities.
        
        Args:
            query: Search query
            entity_types: Filter by entity types
            limit: Maximum results
        
        Returns:
            List of search results with scores
        """
        # Use fulltext index
        cypher = """
        CALL db.index.fulltext.queryNodes('entity_search', $query)
        YIELD node, score
        WHERE $entity_types IS NULL OR node.entity_type IN $entity_types
        RETURN node, score
        ORDER BY score DESC
        LIMIT $limit
        """
        
        results = []
        async with self.async_driver.session(database=self.database) as session:
            result = await session.run(
                cypher,
                query=query,
                entity_types=entity_types,
                limit=limit,
            )
            
            async for record in result:
                node = record["node"]
                entity = Entity(
                    entity_id=node.get("entity_id", ""),
                    entity_type=node.get("entity_type", "Entity"),
                    name=node.get("name", ""),
                    properties=dict(node),
                )
                results.append(GraphSearchResult(
                    entity=entity,
                    score=record["score"],
                ))
        
        return results
    
    async def get_document_context(
        self,
        chunk_id: str,
        context_hops: int = 2,
    ) -> dict[str, Any]:
        """
        Get context for a document chunk from the knowledge graph.
        
        Args:
            chunk_id: Chunk entity ID
            context_hops: Number of relationship hops
        
        Returns:
            Dictionary with related entities and context
        """
        query = f"""
        MATCH (chunk {{entity_id: $chunk_id}})
        OPTIONAL MATCH (chunk)-[r*1..{context_hops}]-(related)
        WITH chunk, collect(DISTINCT related) as related_nodes, collect(DISTINCT r) as rels
        RETURN chunk, related_nodes, size(related_nodes) as related_count
        """
        
        async with self.async_driver.session(database=self.database) as session:
            result = await session.run(query, chunk_id=chunk_id)
            record = await result.single()
            
            if not record:
                return {"chunk_id": chunk_id, "related": [], "context": ""}
            
            related = []
            for node in record["related_nodes"]:
                if node:
                    related.append({
                        "entity_id": node.get("entity_id", ""),
                        "name": node.get("name", ""),
                        "type": node.get("entity_type", ""),
                    })
            
            return {
                "chunk_id": chunk_id,
                "related": related,
                "related_count": record["related_count"],
            }
    
    async def delete_entity(self, entity_id: str) -> None:
        """Delete an entity and its relationships."""
        query = """
        MATCH (e {entity_id: $entity_id})
        DETACH DELETE e
        """
        
        async with self.async_driver.session(database=self.database) as session:
            await session.run(query, entity_id=entity_id)
    
    async def delete_document(self, doc_id: str) -> None:
        """Delete a document and all its chunks from the graph."""
        query = """
        MATCH (d:Document {entity_id: $doc_id})
        OPTIONAL MATCH (d)-[:CONTAINS]->(chunk:Chunk)
        DETACH DELETE d, chunk
        """
        
        async with self.async_driver.session(database=self.database) as session:
            await session.run(query, doc_id=doc_id)
    
    async def get_statistics(self) -> dict[str, Any]:
        """Get graph statistics."""
        query = """
        MATCH (n)
        WITH labels(n) as labels, count(*) as count
        RETURN labels, count
        ORDER BY count DESC
        """
        
        stats = {"node_counts": {}, "total_nodes": 0}
        
        async with self.async_driver.session(database=self.database) as session:
            result = await session.run(query)
            
            async for record in result:
                label = record["labels"][0] if record["labels"] else "Unknown"
                count = record["count"]
                stats["node_counts"][label] = count
                stats["total_nodes"] += count
            
            # Count relationships
            rel_result = await session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_record = await rel_result.single()
            stats["total_relationships"] = rel_record["count"] if rel_record else 0
        
        return stats
    
    async def check_health(self) -> bool:
        """Check if Neo4j is accessible."""
        try:
            async with self.async_driver.session(database=self.database) as session:
                await session.run("RETURN 1")
            return True
        except ServiceUnavailable:
            return False
    
    async def close(self) -> None:
        """Close the driver connections."""
        if self._async_driver:
            await self._async_driver.close()
            self._async_driver = None
        if self._driver:
            self._driver.close()
            self._driver = None


class EntityExtractor:
    """
    Extract entities from text using NER.
    
    Uses spaCy or transformers for named entity recognition.
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        self._nlp = None
    
    def _load_model(self):
        """Lazy load spaCy model."""
        if self._nlp is None:
            import spacy
            try:
                self._nlp = spacy.load(self.model_name)
            except OSError:
                logger.info(f"Downloading spaCy model: {self.model_name}")
                from spacy.cli import download
                download(self.model_name)
                self._nlp = spacy.load(self.model_name)
        return self._nlp
    
    def extract(self, text: str, doc_id: str = "") -> list[Entity]:
        """
        Extract entities from text.
        
        Args:
            text: Input text
            doc_id: Document ID for entity IDs
        
        Returns:
            List of extracted entities
        """
        nlp = self._load_model()
        doc = nlp(text)
        
        entities = []
        seen = set()
        
        for ent in doc.ents:
            # Map spaCy entity types to our types
            type_map = {
                "PERSON": EntityType.PERSON,
                "ORG": EntityType.ORGANIZATION,
                "GPE": EntityType.LOCATION,
                "LOC": EntityType.LOCATION,
                "DATE": EntityType.DATE,
                "EVENT": EntityType.EVENT,
                "PRODUCT": EntityType.PRODUCT,
                "LAW": EntityType.REGULATION,
            }
            
            entity_type = type_map.get(ent.label_, EntityType.CONCEPT)
            
            # Deduplicate by name
            key = (ent.text.lower(), entity_type)
            if key in seen:
                continue
            seen.add(key)
            
            entity_id = f"{doc_id}-{entity_type.value.lower()}-{len(entities)}"
            
            entities.append(Entity(
                entity_id=entity_id,
                entity_type=entity_type,
                name=ent.text,
                properties={
                    "source_doc": doc_id,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                    "label": ent.label_,
                },
            ))
        
        return entities


# Singleton instance
_kg_client: KnowledgeGraphClient | None = None


def get_knowledge_graph_client() -> KnowledgeGraphClient:
    """Get singleton knowledge graph client."""
    global _kg_client
    if _kg_client is None:
        _kg_client = KnowledgeGraphClient()
    return _kg_client


__all__ = [
    "EntityType",
    "RelationType",
    "Entity",
    "Relationship",
    "GraphSearchResult",
    "KnowledgeGraphClient",
    "EntityExtractor",
    "get_knowledge_graph_client",
]
