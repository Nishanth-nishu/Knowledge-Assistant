"""
Middleware Package.

Provides FastAPI middleware for:
- JWT authentication
- API key authentication
- Rate limiting
"""

from src.api.middleware.auth import (
    TokenData,
    UserClaims,
    AuthResult,
    JWTManager,
    APIKeyManager,
    get_jwt_manager,
    get_api_key_manager,
    authenticate,
    require_auth,
    require_admin,
    require_scope,
    hash_password,
    verify_password,
)
from src.api.middleware.rate_limit import (
    RateLimitConfig,
    RateLimitResult,
    TokenBucket,
    SlidingWindowCounter,
    InMemoryRateLimiter,
    RedisRateLimiter,
    RateLimitMiddleware,
    create_rate_limiter,
)

__all__ = [
    # Auth
    "TokenData",
    "UserClaims",
    "AuthResult",
    "JWTManager",
    "APIKeyManager",
    "get_jwt_manager",
    "get_api_key_manager",
    "authenticate",
    "require_auth",
    "require_admin",
    "require_scope",
    "hash_password",
    "verify_password",
    # Rate Limit
    "RateLimitConfig",
    "RateLimitResult",
    "TokenBucket",
    "SlidingWindowCounter",
    "InMemoryRateLimiter",
    "RedisRateLimiter",
    "RateLimitMiddleware",
    "create_rate_limiter",
]
