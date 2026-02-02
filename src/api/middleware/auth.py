"""
JWT Authentication Module.

Provides:
- JWT token creation and validation
- User authentication
- API key support
- Role-based access control
"""

import logging
from datetime import datetime, timedelta
from typing import Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field

from src.config import get_security_settings

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class TokenData(BaseModel):
    """JWT token payload data."""
    sub: str  # Subject (user ID)
    exp: datetime
    iat: datetime
    roles: list[str] = Field(default_factory=list)
    scopes: list[str] = Field(default_factory=list)


class UserClaims(BaseModel):
    """User claims extracted from token."""
    user_id: str
    roles: list[str] = Field(default_factory=list)
    scopes: list[str] = Field(default_factory=list)
    is_admin: bool = False


class AuthResult(BaseModel):
    """Authentication result."""
    authenticated: bool
    user: UserClaims | None = None
    method: str = ""  # "jwt", "api_key", "anonymous"
    error: str | None = None


class JWTManager:
    """
    JWT token management.
    
    Handles creation, validation, and refresh of JWT tokens.
    """
    
    def __init__(
        self,
        secret_key: str | None = None,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 7,
    ):
        """
        Initialize JWT manager.
        
        Args:
            secret_key: Secret key for signing tokens
            algorithm: JWT algorithm
            access_token_expire_minutes: Access token expiry
            refresh_token_expire_days: Refresh token expiry
        """
        settings = get_security_settings()
        
        self.secret_key = secret_key or (
            settings.jwt_secret_key.get_secret_value() 
            if settings.jwt_secret_key else "dev-secret-key-change-in-production"
        )
        self.algorithm = algorithm or settings.jwt_algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
    
    def create_access_token(
        self,
        user_id: str,
        roles: list[str] | None = None,
        scopes: list[str] | None = None,
        expires_delta: timedelta | None = None,
    ) -> str:
        """
        Create a JWT access token.
        
        Args:
            user_id: User identifier
            roles: User roles
            scopes: Permission scopes
            expires_delta: Custom expiration time
        
        Returns:
            Encoded JWT token
        """
        now = datetime.utcnow()
        
        if expires_delta:
            expire = now + expires_delta
        else:
            expire = now + timedelta(minutes=self.access_token_expire_minutes)
        
        payload = {
            "sub": user_id,
            "iat": now,
            "exp": expire,
            "roles": roles or [],
            "scopes": scopes or [],
            "type": "access",
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(
        self,
        user_id: str,
        expires_delta: timedelta | None = None,
    ) -> str:
        """
        Create a JWT refresh token.
        
        Args:
            user_id: User identifier
            expires_delta: Custom expiration time
        
        Returns:
            Encoded JWT refresh token
        """
        now = datetime.utcnow()
        
        if expires_delta:
            expire = now + expires_delta
        else:
            expire = now + timedelta(days=self.refresh_token_expire_days)
        
        payload = {
            "sub": user_id,
            "iat": now,
            "exp": expire,
            "type": "refresh",
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> TokenData | None:
        """
        Verify and decode a JWT token.
        
        Args:
            token: JWT token string
        
        Returns:
            TokenData if valid, None otherwise
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
            )
            
            return TokenData(
                sub=payload.get("sub", ""),
                exp=datetime.fromtimestamp(payload.get("exp", 0)),
                iat=datetime.fromtimestamp(payload.get("iat", 0)),
                roles=payload.get("roles", []),
                scopes=payload.get("scopes", []),
            )
        except JWTError as e:
            logger.warning(f"JWT verification failed: {e}")
            return None
    
    def refresh_access_token(self, refresh_token: str) -> str | None:
        """
        Create new access token from refresh token.
        
        Args:
            refresh_token: Valid refresh token
        
        Returns:
            New access token or None if invalid
        """
        try:
            payload = jwt.decode(
                refresh_token,
                self.secret_key,
                algorithms=[self.algorithm],
            )
            
            if payload.get("type") != "refresh":
                return None
            
            user_id = payload.get("sub")
            if not user_id:
                return None
            
            # Create new access token
            return self.create_access_token(user_id=user_id)
            
        except JWTError:
            return None


class APIKeyManager:
    """
    API key management and validation.
    
    Supports both static keys and database-backed keys.
    """
    
    def __init__(self, valid_keys: dict[str, dict] | None = None):
        """
        Initialize API key manager.
        
        Args:
            valid_keys: Dict mapping API keys to their metadata
                       {key: {"user_id": "", "scopes": [], "rate_limit": 100}}
        """
        settings = get_security_settings()
        
        self._valid_keys = valid_keys or {}
        
        # Add default API keys from settings (for development)
        if settings.api_keys:
            for i, key in enumerate(settings.api_keys):
                self._valid_keys[key.get_secret_value()] = {
                    "user_id": f"api-user-{i}",
                    "scopes": ["read", "write"],
                    "rate_limit": 100,
                }
    
    def validate_key(self, api_key: str) -> dict | None:
        """
        Validate an API key.
        
        Args:
            api_key: API key to validate
        
        Returns:
            Key metadata if valid, None otherwise
        """
        return self._valid_keys.get(api_key)
    
    def add_key(
        self,
        api_key: str,
        user_id: str,
        scopes: list[str] | None = None,
        rate_limit: int = 100,
    ) -> None:
        """Add a new API key."""
        self._valid_keys[api_key] = {
            "user_id": user_id,
            "scopes": scopes or ["read"],
            "rate_limit": rate_limit,
        }
    
    def revoke_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        if api_key in self._valid_keys:
            del self._valid_keys[api_key]
            return True
        return False


# Singleton instances
_jwt_manager: JWTManager | None = None
_api_key_manager: APIKeyManager | None = None


def get_jwt_manager() -> JWTManager:
    """Get singleton JWT manager."""
    global _jwt_manager
    if _jwt_manager is None:
        _jwt_manager = JWTManager()
    return _jwt_manager


def get_api_key_manager() -> APIKeyManager:
    """Get singleton API key manager."""
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager


async def authenticate(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    api_key: str | None = Depends(api_key_header),
) -> AuthResult:
    """
    Authenticate request via JWT or API key.
    
    Args:
        credentials: Bearer token credentials
        api_key: API key from header
    
    Returns:
        AuthResult with authentication status
    """
    # Try JWT first
    if credentials and credentials.credentials:
        jwt_manager = get_jwt_manager()
        token_data = jwt_manager.verify_token(credentials.credentials)
        
        if token_data:
            return AuthResult(
                authenticated=True,
                user=UserClaims(
                    user_id=token_data.sub,
                    roles=token_data.roles,
                    scopes=token_data.scopes,
                    is_admin="admin" in token_data.roles,
                ),
                method="jwt",
            )
        else:
            return AuthResult(
                authenticated=False,
                error="Invalid or expired token",
            )
    
    # Try API key
    if api_key:
        api_key_manager = get_api_key_manager()
        key_data = api_key_manager.validate_key(api_key)
        
        if key_data:
            return AuthResult(
                authenticated=True,
                user=UserClaims(
                    user_id=key_data["user_id"],
                    scopes=key_data.get("scopes", []),
                    is_admin=False,
                ),
                method="api_key",
            )
        else:
            return AuthResult(
                authenticated=False,
                error="Invalid API key",
            )
    
    # No credentials provided
    return AuthResult(
        authenticated=False,
        error="No authentication credentials provided",
    )


async def require_auth(
    auth: AuthResult = Depends(authenticate),
) -> UserClaims:
    """
    Dependency that requires authentication.
    
    Raises 401 if not authenticated.
    """
    if not auth.authenticated:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=auth.error or "Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return auth.user


async def require_admin(
    user: UserClaims = Depends(require_auth),
) -> UserClaims:
    """
    Dependency that requires admin role.
    
    Raises 403 if not admin.
    """
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    
    return user


def require_scope(required_scope: str):
    """
    Factory for scope-checking dependency.
    
    Args:
        required_scope: Scope that must be present
    
    Returns:
        Dependency function
    """
    async def check_scope(user: UserClaims = Depends(require_auth)) -> UserClaims:
        if required_scope not in user.scopes and not user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Scope '{required_scope}' required",
            )
        return user
    
    return check_scope


# Utility functions
def hash_password(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


__all__ = [
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
]
