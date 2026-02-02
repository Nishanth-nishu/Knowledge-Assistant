"""
Tests for API security middleware.
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from src.api.middleware.auth import (
    JWTManager,
    APIKeyManager,
    UserClaims,
    hash_password,
    verify_password,
)
from src.api.middleware.rate_limit import (
    TokenBucket,
    SlidingWindowCounter,
    InMemoryRateLimiter,
    RateLimitConfig,
)


class TestJWTManager:
    """Tests for JWT token management."""
    
    def test_create_access_token(self):
        """Test access token creation."""
        manager = JWTManager(secret_key="test-secret")
        
        token = manager.create_access_token(
            user_id="user123",
            roles=["user", "admin"],
            scopes=["read", "write"],
        )
        
        assert token is not None
        assert len(token) > 0
    
    def test_verify_token(self):
        """Test token verification."""
        manager = JWTManager(secret_key="test-secret")
        
        token = manager.create_access_token(
            user_id="user123",
            roles=["user"],
            scopes=["read"],
        )
        
        data = manager.verify_token(token)
        
        assert data is not None
        assert data.sub == "user123"
        assert "user" in data.roles
        assert "read" in data.scopes
    
    def test_verify_invalid_token(self):
        """Test verification of invalid token."""
        manager = JWTManager(secret_key="test-secret")
        
        data = manager.verify_token("invalid-token")
        
        assert data is None
    
    def test_verify_wrong_secret(self):
        """Test verification with wrong secret."""
        manager1 = JWTManager(secret_key="secret1")
        manager2 = JWTManager(secret_key="secret2")
        
        token = manager1.create_access_token(user_id="user123")
        data = manager2.verify_token(token)
        
        assert data is None
    
    def test_refresh_token(self):
        """Test refresh token creation and use."""
        manager = JWTManager(secret_key="test-secret")
        
        refresh = manager.create_refresh_token(user_id="user123")
        new_access = manager.refresh_access_token(refresh)
        
        assert new_access is not None
        
        data = manager.verify_token(new_access)
        assert data.sub == "user123"
    
    def test_expired_token(self):
        """Test expired token is rejected."""
        manager = JWTManager(secret_key="test-secret")
        
        # Create token that expires immediately
        token = manager.create_access_token(
            user_id="user123",
            expires_delta=timedelta(seconds=-1),
        )
        
        data = manager.verify_token(token)
        assert data is None


class TestAPIKeyManager:
    """Tests for API key management."""
    
    def test_validate_key(self):
        """Test API key validation."""
        manager = APIKeyManager(valid_keys={
            "test-key-123": {
                "user_id": "api-user",
                "scopes": ["read"],
                "rate_limit": 100,
            }
        })
        
        result = manager.validate_key("test-key-123")
        
        assert result is not None
        assert result["user_id"] == "api-user"
    
    def test_invalid_key(self):
        """Test invalid key returns None."""
        manager = APIKeyManager()
        
        result = manager.validate_key("invalid-key")
        
        assert result is None
    
    def test_add_and_revoke_key(self):
        """Test adding and revoking keys."""
        manager = APIKeyManager()
        
        manager.add_key("new-key", user_id="user1", scopes=["write"])
        
        result = manager.validate_key("new-key")
        assert result is not None
        
        manager.revoke_key("new-key")
        
        result = manager.validate_key("new-key")
        assert result is None


class TestPasswordHashing:
    """Tests for password utilities."""
    
    def test_hash_password(self):
        """Test password hashing."""
        password = "secure123!"
        hashed = hash_password(password)
        
        assert hashed != password
        assert len(hashed) > 0
    
    def test_verify_password(self):
        """Test password verification."""
        password = "secure123!"
        hashed = hash_password(password)
        
        assert verify_password(password, hashed) is True
        assert verify_password("wrong", hashed) is False


class TestTokenBucket:
    """Tests for token bucket rate limiter."""
    
    def test_consume_within_capacity(self):
        """Test consuming tokens within capacity."""
        bucket = TokenBucket(rate=10, capacity=10)
        
        assert bucket.consume(5) is True
        assert bucket.remaining == 5
    
    def test_consume_exceeds_capacity(self):
        """Test consuming more than available tokens."""
        bucket = TokenBucket(rate=1, capacity=5)
        
        # Use all tokens
        for _ in range(5):
            assert bucket.consume() is True
        
        # Next should fail
        assert bucket.consume() is False
    
    def test_token_refill(self):
        """Test tokens refill over time."""
        bucket = TokenBucket(rate=100, capacity=10)  # Fast refill
        
        # Use all tokens
        for _ in range(10):
            bucket.consume()
        
        # Wait a bit
        time.sleep(0.1)
        
        # Should have refilled
        assert bucket.consume() is True


class TestSlidingWindowCounter:
    """Tests for sliding window rate limiter."""
    
    def test_allows_within_limit(self):
        """Test requests within limit are allowed."""
        counter = SlidingWindowCounter(max_requests=10, window_seconds=60)
        
        for _ in range(10):
            result = counter.is_allowed()
            assert result.allowed is True
        
        assert counter.is_allowed().allowed is False
    
    def test_remaining_count(self):
        """Test remaining count is accurate."""
        counter = SlidingWindowCounter(max_requests=10, window_seconds=60)
        
        for i in range(5):
            result = counter.is_allowed()
        
        assert result.remaining == 5


class TestInMemoryRateLimiter:
    """Tests for in-memory rate limiter."""
    
    def test_allows_requests(self):
        """Test rate limiter allows requests."""
        config = RateLimitConfig(
            requests_per_minute=10,
            requests_per_hour=100,
            burst_size=5,
        )
        limiter = InMemoryRateLimiter(config)
        
        result = limiter.is_allowed("test-key")
        
        assert result.allowed is True
        assert result.remaining > 0
    
    def test_blocks_after_limit(self):
        """Test rate limiter blocks after limit."""
        config = RateLimitConfig(
            requests_per_minute=5,
            requests_per_hour=100,
            burst_size=5,
        )
        limiter = InMemoryRateLimiter(config)
        
        # Exhaust burst limit
        for _ in range(5):
            limiter.is_allowed("test-key")
        
        result = limiter.is_allowed("test-key")
        assert result.allowed is False
    
    def test_separate_keys(self):
        """Test separate keys have separate limits."""
        config = RateLimitConfig(
            requests_per_minute=2,
            burst_size=2,
        )
        limiter = InMemoryRateLimiter(config)
        
        # Use up key1's limit
        limiter.is_allowed("key1")
        limiter.is_allowed("key1")
        
        # key2 should still have quota
        result = limiter.is_allowed("key2")
        assert result.allowed is True
    
    def test_disabled_limiter(self):
        """Test disabled rate limiter allows all."""
        config = RateLimitConfig(enabled=False)
        limiter = InMemoryRateLimiter(config)
        
        for _ in range(100):
            result = limiter.is_allowed("test-key")
            assert result.allowed is True
