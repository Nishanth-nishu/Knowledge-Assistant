"""
Rate Limiting Middleware.

Provides:
- Token bucket rate limiting
- Per-user and per-IP limits
- Sliding window algorithm
- Redis-backed distributed rate limiting
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_size: int = 10
    enabled: bool = True


@dataclass
class RateLimitResult:
    """Result of rate limit check."""
    allowed: bool
    remaining: int
    reset_at: float  # Unix timestamp
    retry_after: int | None = None  # Seconds


class TokenBucket:
    """
    Token bucket rate limiter.
    
    Allows bursting while enforcing average rate.
    """
    
    def __init__(
        self,
        rate: float,
        capacity: int,
    ):
        """
        Initialize token bucket.
        
        Args:
            rate: Tokens added per second
            capacity: Maximum tokens (burst size)
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens.
        
        Args:
            tokens: Number of tokens to consume
        
        Returns:
            True if tokens were available
        """
        now = time.time()
        
        # Add new tokens based on time passed
        elapsed = now - self.last_update
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.rate
        )
        self.last_update = now
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        
        return False
    
    @property
    def remaining(self) -> int:
        """Get remaining tokens."""
        return int(self.tokens)
    
    @property
    def time_to_refill(self) -> float:
        """Time until bucket is full."""
        tokens_needed = self.capacity - self.tokens
        return tokens_needed / self.rate if self.rate > 0 else 0


class SlidingWindowCounter:
    """
    Sliding window rate limiter.
    
    More accurate than fixed windows, smoother rate limiting.
    """
    
    def __init__(
        self,
        max_requests: int,
        window_seconds: int,
    ):
        """
        Initialize sliding window.
        
        Args:
            max_requests: Max requests per window
            window_seconds: Window size in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: list[float] = []  # Timestamps
    
    def is_allowed(self) -> RateLimitResult:
        """
        Check if request is allowed.
        
        Returns:
            RateLimitResult with status
        """
        now = time.time()
        window_start = now - self.window_seconds
        
        # Remove old requests outside window
        self.requests = [t for t in self.requests if t > window_start]
        
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return RateLimitResult(
                allowed=True,
                remaining=self.max_requests - len(self.requests),
                reset_at=now + self.window_seconds,
            )
        
        # Calculate when oldest request expires
        oldest = min(self.requests) if self.requests else now
        retry_after = int(oldest + self.window_seconds - now) + 1
        
        return RateLimitResult(
            allowed=False,
            remaining=0,
            reset_at=oldest + self.window_seconds,
            retry_after=retry_after,
        )


class InMemoryRateLimiter:
    """
    In-memory rate limiter.
    
    Suitable for single-instance deployments.
    For distributed systems, use RedisRateLimiter.
    """
    
    def __init__(self, config: RateLimitConfig | None = None):
        """
        Initialize rate limiter.
        
        Args:
            config: Rate limit configuration
        """
        self.config = config or RateLimitConfig()
        
        # Per-key rate limiters
        self._minute_windows: dict[str, SlidingWindowCounter] = {}
        self._hour_windows: dict[str, SlidingWindowCounter] = {}
        self._buckets: dict[str, TokenBucket] = {}
        
        # Cleanup task
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.time()
    
    def _get_or_create_limiters(self, key: str) -> tuple[
        SlidingWindowCounter, SlidingWindowCounter, TokenBucket
    ]:
        """Get or create rate limiters for a key."""
        if key not in self._minute_windows:
            self._minute_windows[key] = SlidingWindowCounter(
                max_requests=self.config.requests_per_minute,
                window_seconds=60,
            )
        
        if key not in self._hour_windows:
            self._hour_windows[key] = SlidingWindowCounter(
                max_requests=self.config.requests_per_hour,
                window_seconds=3600,
            )
        
        if key not in self._buckets:
            self._buckets[key] = TokenBucket(
                rate=self.config.requests_per_minute / 60,
                capacity=self.config.burst_size,
            )
        
        return (
            self._minute_windows[key],
            self._hour_windows[key],
            self._buckets[key],
        )
    
    def is_allowed(self, key: str) -> RateLimitResult:
        """
        Check if request is allowed for key.
        
        Args:
            key: Rate limit key (user ID, IP, etc.)
        
        Returns:
            RateLimitResult with status
        """
        if not self.config.enabled:
            return RateLimitResult(allowed=True, remaining=999, reset_at=0)
        
        minute_window, hour_window, bucket = self._get_or_create_limiters(key)
        
        # Check all limits
        minute_result = minute_window.is_allowed()
        if not minute_result.allowed:
            return minute_result
        
        hour_result = hour_window.is_allowed()
        if not hour_result.allowed:
            return hour_result
        
        # Token bucket for burst control
        if not bucket.consume():
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_at=time.time() + 1,
                retry_after=1,
            )
        
        return RateLimitResult(
            allowed=True,
            remaining=min(minute_result.remaining, bucket.remaining),
            reset_at=minute_result.reset_at,
        )
    
    def cleanup(self) -> None:
        """Remove old rate limit entries."""
        now = time.time()
        
        if now - self._last_cleanup < self._cleanup_interval:
            return
        
        # Remove stale entries (no requests in 2x window)
        stale_threshold = now - 7200  # 2 hours
        
        for key, window in list(self._minute_windows.items()):
            if not window.requests or max(window.requests) < stale_threshold:
                del self._minute_windows[key]
                self._hour_windows.pop(key, None)
                self._buckets.pop(key, None)
        
        self._last_cleanup = now
        logger.debug(f"Rate limiter cleanup: {len(self._minute_windows)} active keys")


class RedisRateLimiter:
    """
    Redis-backed distributed rate limiter.
    
    Uses Redis for rate limit tracking across instances.
    """
    
    def __init__(
        self,
        redis_client=None,
        config: RateLimitConfig | None = None,
        key_prefix: str = "rl:",
    ):
        """
        Initialize Redis rate limiter.
        
        Args:
            redis_client: Redis async client
            config: Rate limit configuration
            key_prefix: Prefix for Redis keys
        """
        self.redis = redis_client
        self.config = config or RateLimitConfig()
        self.key_prefix = key_prefix
    
    async def _get_redis(self):
        """Get Redis client with lazy initialization."""
        if self.redis is None:
            import redis.asyncio as redis
            settings = get_settings()
            self.redis = redis.from_url(settings.redis.url)
        return self.redis
    
    async def is_allowed(self, key: str) -> RateLimitResult:
        """
        Check if request is allowed (async).
        
        Args:
            key: Rate limit key
        
        Returns:
            RateLimitResult with status
        """
        if not self.config.enabled:
            return RateLimitResult(allowed=True, remaining=999, reset_at=0)
        
        redis_client = await self._get_redis()
        
        now = time.time()
        minute_key = f"{self.key_prefix}{key}:min"
        hour_key = f"{self.key_prefix}{key}:hour"
        
        pipe = redis_client.pipeline()
        
        # Minute window
        pipe.zadd(minute_key, {str(now): now})
        pipe.zremrangebyscore(minute_key, 0, now - 60)
        pipe.zcard(minute_key)
        pipe.expire(minute_key, 120)
        
        # Hour window
        pipe.zadd(hour_key, {str(now): now})
        pipe.zremrangebyscore(hour_key, 0, now - 3600)
        pipe.zcard(hour_key)
        pipe.expire(hour_key, 7200)
        
        results = await pipe.execute()
        
        minute_count = results[2]
        hour_count = results[6]
        
        # Check limits
        if minute_count > self.config.requests_per_minute:
            # Get oldest request in window
            oldest = await redis_client.zrange(minute_key, 0, 0, withscores=True)
            retry_after = int(60 - (now - oldest[0][1])) + 1 if oldest else 60
            
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_at=now + 60,
                retry_after=retry_after,
            )
        
        if hour_count > self.config.requests_per_hour:
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_at=now + 3600,
                retry_after=60,
            )
        
        return RateLimitResult(
            allowed=True,
            remaining=self.config.requests_per_minute - minute_count,
            reset_at=now + 60,
        )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI rate limiting middleware.
    
    Applies rate limits per user/IP to all requests.
    """
    
    def __init__(
        self,
        app,
        limiter: InMemoryRateLimiter | RedisRateLimiter | None = None,
        config: RateLimitConfig | None = None,
        key_func: Callable[[Request], str] | None = None,
        exclude_paths: list[str] | None = None,
    ):
        """
        Initialize middleware.
        
        Args:
            app: FastAPI application
            limiter: Rate limiter instance
            config: Rate limit config
            key_func: Function to extract rate limit key from request
            exclude_paths: Paths to exclude from rate limiting
        """
        super().__init__(app)
        
        self.limiter = limiter or InMemoryRateLimiter(config)
        self.key_func = key_func or self._default_key_func
        self.exclude_paths = set(exclude_paths or ["/health", "/ready", "/metrics"])
    
    def _default_key_func(self, request: Request) -> str:
        """Default key extraction: use user ID or IP."""
        # Try to get user from state (set by auth middleware)
        user = getattr(request.state, "user", None)
        if user:
            return f"user:{user.user_id}"
        
        # Fall back to IP
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"
        
        return f"ip:{ip}"
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with rate limiting."""
        # Skip excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Get rate limit key
        key = self.key_func(request)
        
        # Check rate limit
        if isinstance(self.limiter, RedisRateLimiter):
            result = await self.limiter.is_allowed(key)
        else:
            result = self.limiter.is_allowed(key)
            # Periodic cleanup
            self.limiter.cleanup()
        
        if not result.allowed:
            logger.warning(f"Rate limit exceeded for {key}")
            
            return Response(
                content='{"detail": "Rate limit exceeded"}',
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                media_type="application/json",
                headers={
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(result.reset_at)),
                    "Retry-After": str(result.retry_after or 60),
                },
            )
        
        # Add rate limit headers to response
        response = await call_next(request)
        
        response.headers["X-RateLimit-Remaining"] = str(result.remaining)
        response.headers["X-RateLimit-Reset"] = str(int(result.reset_at))
        
        return response


# Factory functions
def create_rate_limiter(
    use_redis: bool = False,
    config: RateLimitConfig | None = None,
) -> InMemoryRateLimiter | RedisRateLimiter:
    """Create a rate limiter instance."""
    if use_redis:
        return RedisRateLimiter(config=config)
    return InMemoryRateLimiter(config=config)


__all__ = [
    "RateLimitConfig",
    "RateLimitResult",
    "TokenBucket",
    "SlidingWindowCounter",
    "InMemoryRateLimiter",
    "RedisRateLimiter",
    "RateLimitMiddleware",
    "create_rate_limiter",
]
