"""
Configuration management for the Enterprise Knowledge Assistant.

Uses pydantic-settings for type-safe configuration with environment variable support.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """LLM configuration supporting any OpenAI-compatible API."""
    
    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        extra="ignore",
    )
    
    # Provider selection
    provider: Literal["openai", "azure", "ollama", "anthropic", "custom"] = "openai"
    
    # API configuration
    api_base: str = "https://api.openai.com/v1"
    api_key: SecretStr = Field(default=SecretStr(""))
    model: str = "gpt-4-turbo"
    
    # Azure-specific
    api_version: str | None = None
    azure_deployment: str | None = None
    
    # Generation parameters
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=128000)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    
    # Client settings
    timeout: int = Field(default=60, ge=1)
    max_retries: int = Field(default=3, ge=0)
    
    @field_validator("api_base")
    @classmethod
    def normalize_api_base(cls, v: str) -> str:
        """Ensure API base URL doesn't have trailing slash."""
        return v.rstrip("/")


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        env_file=".env",
        extra="ignore",
    )
    
    model: str = "BAAI/bge-m3"
    device: Literal["cuda", "cpu", "mps"] = "cuda"
    batch_size: int = Field(default=32, ge=1)
    max_length: int = Field(default=512, ge=1)
    normalize: bool = True
    
    # SPLADE sparse embeddings
    splade_model: str = Field(
        default="naver/splade-cocondenser-ensembledistil",
        alias="SPLADE_MODEL"
    )
    splade_device: Literal["cuda", "cpu", "mps"] = Field(
        default="cuda",
        alias="SPLADE_DEVICE"
    )


class RerankerSettings(BaseSettings):
    """Reranker model configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="RERANKER_",
        env_file=".env",
        extra="ignore",
    )
    
    model: str = "colbert-ir/colbertv2.0"
    device: Literal["cuda", "cpu", "mps"] = "cuda"
    top_k: int = Field(default=10, ge=1)
    batch_size: int = Field(default=16, ge=1)


class QdrantSettings(BaseSettings):
    """Qdrant vector database configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="QDRANT_",
        env_file=".env",
        extra="ignore",
    )
    
    url: str = "http://localhost:6333"
    api_key: SecretStr | None = None
    collection_name: str = "enterprise_docs"
    vector_size: int = 1024
    distance: Literal["Cosine", "Euclid", "Dot"] = "Cosine"
    
    # Performance settings
    prefer_grpc: bool = True
    timeout: int = 30


class Neo4jSettings(BaseSettings):
    """Neo4j graph database configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="NEO4J_",
        env_file=".env",
        extra="ignore",
    )
    
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: SecretStr = Field(default=SecretStr(""))
    database: str = "neo4j"
    
    # Connection settings
    max_connection_lifetime: int = 3600
    max_connection_pool_size: int = 50


class RedisSettings(BaseSettings):
    """Redis cache configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="REDIS_",
        env_file=".env",
        extra="ignore",
    )
    
    url: str = "redis://localhost:6379/0"
    password: SecretStr | None = None
    ttl_seconds: int = Field(default=3600, ge=0)
    max_connections: int = Field(default=10, ge=1)


class RetrievalSettings(BaseSettings):
    """Retrieval pipeline configuration."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )
    
    # Hybrid search weights (should sum to 1.0)
    bm25_weight: float = Field(default=0.3, ge=0.0, le=1.0, alias="BM25_WEIGHT")
    dense_weight: float = Field(default=0.4, ge=0.0, le=1.0, alias="DENSE_WEIGHT")
    sparse_weight: float = Field(default=0.3, ge=0.0, le=1.0, alias="SPARSE_WEIGHT")
    
    # RRF parameter
    rrf_k: int = Field(default=60, ge=1, alias="RRF_K")
    
    # Search settings
    default_top_k: int = Field(default=20, ge=1, alias="DEFAULT_TOP_K")
    rerank_top_k: int = Field(default=10, ge=1, alias="RERANK_TOP_K")
    
    @field_validator("dense_weight")
    @classmethod
    def validate_weights_sum(cls, v: float, info) -> float:
        """Weights should approximately sum to 1.0."""
        # Note: Full validation would require all values
        return v


class SecuritySettings(BaseSettings):
    """Security and authentication configuration."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )
    
    # JWT Configuration
    jwt_secret_key: SecretStr = Field(
        default=SecretStr("change-this-in-production"),
        alias="JWT_SECRET_KEY"
    )
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(
        default=30, alias="JWT_ACCESS_TOKEN_EXPIRE_MINUTES"
    )
    jwt_refresh_token_expire_days: int = Field(
        default=7, alias="JWT_REFRESH_TOKEN_EXPIRE_DAYS"
    )
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100, alias="RATE_LIMIT_REQUESTS")
    rate_limit_window_seconds: int = Field(default=60, alias="RATE_LIMIT_WINDOW_SECONDS")
    
    # API Key
    api_key_header: str = Field(default="X-API-Key", alias="API_KEY_HEADER")


class APISettings(BaseSettings):
    """API server configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="API_",
        env_file=".env",
        extra="ignore",
    )
    
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=4, ge=1)
    reload: bool = False
    log_level: Literal["debug", "info", "warning", "error", "critical"] = "info"
    
    # CORS
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        alias="CORS_ORIGINS"
    )
    cors_allow_credentials: bool = Field(default=True, alias="CORS_ALLOW_CREDENTIALS")


class AgentSettings(BaseSettings):
    """Multi-agent system configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=".env",
        extra="ignore",
    )
    
    max_iterations: int = Field(default=10, ge=1)
    timeout_seconds: int = Field(default=120, ge=1)
    enable_self_validation: bool = True
    enable_multi_hop: bool = True


class DocumentSettings(BaseSettings):
    """Document processing configuration."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )
    
    upload_dir: str = Field(default="./data/uploads", alias="UPLOAD_DIR")
    max_upload_size_mb: int = Field(default=50, ge=1, alias="MAX_UPLOAD_SIZE_MB")
    allowed_extensions: list[str] = Field(
        default=["pdf", "docx", "doc", "txt", "html", "md"],
        alias="ALLOWED_EXTENSIONS"
    )
    
    # Chunking settings
    chunk_size: int = Field(default=512, ge=100, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, ge=0, alias="CHUNK_OVERLAP")
    chunk_strategy: Literal["semantic", "recursive", "fixed"] = Field(
        default="semantic", alias="CHUNK_STRATEGY"
    )


class MonitoringSettings(BaseSettings):
    """Monitoring and observability configuration."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )
    
    prometheus_port: int = Field(default=9090, alias="PROMETHEUS_PORT")
    enable_metrics: bool = Field(default=True, alias="ENABLE_METRICS")
    enable_tracing: bool = Field(default=True, alias="ENABLE_TRACING")
    
    # Jaeger
    jaeger_agent_host: str = Field(default="localhost", alias="JAEGER_AGENT_HOST")
    jaeger_agent_port: int = Field(default=6831, alias="JAEGER_AGENT_PORT")
    
    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: Literal["json", "text"] = Field(default="json", alias="LOG_FORMAT")
    log_file: str | None = Field(default=None, alias="LOG_FILE")
    enable_pii_masking: bool = Field(default=True, alias="ENABLE_PII_MASKING")


class FeatureFlags(BaseSettings):
    """Feature flags for enabling/disabling functionality."""
    
    model_config = SettingsConfigDict(
        env_prefix="FEATURE_",
        env_file=".env",
        extra="ignore",
    )
    
    knowledge_graph: bool = True
    multi_agent: bool = True
    streaming: bool = True
    caching: bool = True


class Settings(BaseSettings):
    """Main application settings aggregating all configuration."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )
    
    # Sub-configurations
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    api: APISettings = Field(default_factory=APISettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    document: DocumentSettings = Field(default_factory=DocumentSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    features: FeatureFlags = Field(default_factory=FeatureFlags)


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Convenience accessors
def get_llm_settings() -> LLMSettings:
    return get_settings().llm


def get_embedding_settings() -> EmbeddingSettings:
    return get_settings().embedding


def get_qdrant_settings() -> QdrantSettings:
    return get_settings().qdrant


def get_neo4j_settings() -> Neo4jSettings:
    return get_settings().neo4j


def get_redis_settings() -> RedisSettings:
    return get_settings().redis
