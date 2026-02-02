"""
Authentication Routes.

Provides endpoints for:
- User login (JWT)
- Token refresh
- API key generation
"""

import logging
from datetime import timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from src.api.middleware.auth import (
    JWTManager,
    UserClaims,
    get_jwt_manager,
    hash_password,
    require_auth,
    verify_password,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


# Request/Response Models
class LoginRequest(BaseModel):
    """Login request model."""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 1800  # 30 minutes


class RefreshRequest(BaseModel):
    """Token refresh request."""
    refresh_token: str


class UserInfo(BaseModel):
    """User info response."""
    user_id: str
    roles: list[str]
    scopes: list[str]


# Mock user store (replace with database in production)
MOCK_USERS = {
    "admin": {
        "password_hash": hash_password("admin123!"),
        "roles": ["admin", "user"],
        "scopes": ["read", "write", "admin"],
    },
    "user": {
        "password_hash": hash_password("user1234!"),
        "roles": ["user"],
        "scopes": ["read"],
    },
}


@router.post("/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    jwt_manager: JWTManager = Depends(get_jwt_manager),
) -> TokenResponse:
    """
    Authenticate user and return JWT tokens.
    
    Args:
        request: Login credentials
        jwt_manager: JWT manager instance
    
    Returns:
        Access and refresh tokens
    """
    # Look up user
    user_data = MOCK_USERS.get(request.username)
    
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )
    
    # Verify password
    if not verify_password(request.password, user_data["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )
    
    # Create tokens
    access_token = jwt_manager.create_access_token(
        user_id=request.username,
        roles=user_data["roles"],
        scopes=user_data["scopes"],
    )
    
    refresh_token = jwt_manager.create_refresh_token(
        user_id=request.username,
    )
    
    logger.info(f"User {request.username} logged in")
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=1800,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: RefreshRequest,
    jwt_manager: JWTManager = Depends(get_jwt_manager),
) -> TokenResponse:
    """
    Refresh access token using refresh token.
    
    Args:
        request: Refresh token
        jwt_manager: JWT manager instance
    
    Returns:
        New access token
    """
    new_access_token = jwt_manager.refresh_access_token(request.refresh_token)
    
    if not new_access_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )
    
    # Create new refresh token as well
    token_data = jwt_manager.verify_token(new_access_token)
    new_refresh_token = jwt_manager.create_refresh_token(
        user_id=token_data.sub if token_data else "",
    )
    
    return TokenResponse(
        access_token=new_access_token,
        refresh_token=new_refresh_token,
        expires_in=1800,
    )


@router.get("/me", response_model=UserInfo)
async def get_current_user(
    user: UserClaims = Depends(require_auth),
) -> UserInfo:
    """
    Get current authenticated user info.
    
    Args:
        user: Authenticated user claims
    
    Returns:
        User information
    """
    return UserInfo(
        user_id=user.user_id,
        roles=user.roles,
        scopes=user.scopes,
    )


@router.post("/logout")
async def logout(
    user: UserClaims = Depends(require_auth),
) -> dict:
    """
    Logout user (client should discard tokens).
    
    In a production system, you'd add the token to a blacklist.
    """
    logger.info(f"User {user.user_id} logged out")
    
    return {"message": "Logged out successfully"}


__all__ = ["router"]
