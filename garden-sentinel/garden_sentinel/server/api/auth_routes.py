"""
Authentication API routes.

Provides endpoints for:
- User login/logout
- Token refresh
- API key management
- User management (admin)
"""

from datetime import timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, EmailStr, Field

from garden_sentinel.server.auth import (
    AuthManager,
    User,
    UserRole,
    APIKey,
    get_auth_manager,
    get_current_user,
    get_authenticated_user,
    AuthenticatedUser,
    require_admin,
    require_operator,
    rate_limit,
)


router = APIRouter(prefix="/auth", tags=["Authentication"])


# ============== Request/Response Models ==============

class LoginRequest(BaseModel):
    """Login credentials."""
    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=1)

    model_config = {
        "json_schema_extra": {
            "examples": [{"username": "admin", "password": "changeme"}]
        }
    }


class LoginResponse(BaseModel):
    """Login response with token."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: dict

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "access_token": "eyJ...",
                "token_type": "bearer",
                "expires_in": 86400,
                "user": {"user_id": "abc123", "username": "admin", "role": "admin"}
            }]
        }
    }


class CreateUserRequest(BaseModel):
    """Create new user request."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    role: str = Field(default="viewer", pattern="^(viewer|operator|admin)$")


class UserResponse(BaseModel):
    """User information response."""
    user_id: str
    username: str
    email: str
    role: str
    created_at: float
    last_login: Optional[float]
    is_active: bool
    mfa_enabled: bool


class ChangePasswordRequest(BaseModel):
    """Change password request."""
    current_password: str
    new_password: str = Field(..., min_length=8)


class CreateAPIKeyRequest(BaseModel):
    """Create API key request."""
    name: str = Field(..., min_length=1, max_length=100)
    device_id: Optional[str] = None
    expires_in_days: Optional[int] = Field(default=None, ge=1, le=365)


class APIKeyResponse(BaseModel):
    """API key response (key only shown once!)."""
    key_id: str
    key: Optional[str] = None  # Only set on creation
    name: str
    device_id: Optional[str]
    role: str
    created_at: float
    expires_at: Optional[float]
    is_active: bool


# ============== Auth Endpoints ==============

@router.post(
    "/login",
    response_model=LoginResponse,
    summary="User login",
    description="Authenticate with username and password to receive a JWT token.",
    responses={
        401: {"description": "Invalid credentials"},
        429: {"description": "Too many login attempts"},
    }
)
async def login(
    request: Request,
    credentials: LoginRequest,
    _: None = Depends(rate_limit),
):
    """
    Authenticate a user and return a JWT token.

    The token should be included in subsequent requests as:
    `Authorization: Bearer <token>`
    """
    auth = get_auth_manager()

    user = auth.authenticate_user(credentials.username, credentials.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    if not user.is_active:
        raise HTTPException(status_code=401, detail="Account is disabled")

    # Create session
    ip_address = request.headers.get("X-Forwarded-For", request.client.host if request.client else None)
    user_agent = request.headers.get("User-Agent")

    token, session = auth.create_session(user, ip_address, user_agent)

    return LoginResponse(
        access_token=token,
        token_type="bearer",
        expires_in=int(auth.token_expiry.total_seconds()),
        user=user.to_dict(),
    )


@router.post(
    "/logout",
    summary="User logout",
    description="Invalidate the current session.",
)
async def logout(
    current_user: User | APIKey = Depends(get_current_user),
):
    """Log out and invalidate the current session."""
    auth = get_auth_manager()

    # If it's a user session, we could invalidate it
    # For now, just return success (client should discard token)
    return {"message": "Logged out successfully"}


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user",
    description="Get information about the currently authenticated user.",
)
async def get_me(
    auth_context: AuthenticatedUser = Depends(get_authenticated_user),
):
    """Get the current user's information."""
    if not auth_context.is_authenticated:
        raise HTTPException(status_code=401, detail="Not authenticated")

    if auth_context.user:
        return UserResponse(**auth_context.user.to_dict())

    # For API key auth, return limited info
    if auth_context.api_key:
        return UserResponse(
            user_id=auth_context.api_key.key_id,
            username=f"device:{auth_context.api_key.name}",
            email="",
            role=auth_context.api_key.role.value,
            created_at=auth_context.api_key.created_at,
            last_login=auth_context.api_key.last_used,
            is_active=auth_context.api_key.is_active,
            mfa_enabled=False,
        )


@router.post(
    "/change-password",
    summary="Change password",
    description="Change the current user's password.",
)
async def change_password(
    request: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
):
    """Change the current user's password."""
    if not isinstance(current_user, User):
        raise HTTPException(status_code=400, detail="API keys cannot change password")

    if not current_user.check_password(request.current_password):
        raise HTTPException(status_code=401, detail="Current password is incorrect")

    auth = get_auth_manager()
    auth.change_password(current_user.user_id, request.new_password)

    return {"message": "Password changed successfully"}


# ============== User Management (Admin) ==============

@router.get(
    "/users",
    response_model=list[UserResponse],
    summary="List all users",
    description="Get a list of all users (admin only).",
    dependencies=[Depends(require_admin())],
)
async def list_users():
    """List all users in the system."""
    auth = get_auth_manager()
    return [UserResponse(**u.to_dict()) for u in auth.users.values()]


@router.post(
    "/users",
    response_model=UserResponse,
    summary="Create user",
    description="Create a new user (admin only).",
    dependencies=[Depends(require_admin())],
)
async def create_user(request: CreateUserRequest):
    """Create a new user."""
    auth = get_auth_manager()

    # Check if username exists
    for user in auth.users.values():
        if user.username == request.username:
            raise HTTPException(status_code=400, detail="Username already exists")

    user = auth.create_user(
        username=request.username,
        email=request.email,
        password=request.password,
        role=UserRole(request.role),
    )

    return UserResponse(**user.to_dict())


@router.delete(
    "/users/{user_id}",
    summary="Delete user",
    description="Delete a user (admin only).",
    dependencies=[Depends(require_admin())],
)
async def delete_user(user_id: str):
    """Delete a user."""
    auth = get_auth_manager()

    if user_id not in auth.users:
        raise HTTPException(status_code=404, detail="User not found")

    # Don't allow deleting the last admin
    user = auth.users[user_id]
    if user.role == UserRole.ADMIN:
        admin_count = sum(1 for u in auth.users.values() if u.role == UserRole.ADMIN)
        if admin_count <= 1:
            raise HTTPException(status_code=400, detail="Cannot delete the last admin user")

    del auth.users[user_id]
    return {"message": "User deleted"}


# ============== API Key Management ==============

@router.get(
    "/api-keys",
    response_model=list[APIKeyResponse],
    summary="List API keys",
    description="List all API keys (operator or admin).",
    dependencies=[Depends(require_operator())],
)
async def list_api_keys():
    """List all API keys."""
    auth = get_auth_manager()
    return [
        APIKeyResponse(
            key_id=k.key_id,
            name=k.name,
            device_id=k.device_id,
            role=k.role.value,
            created_at=k.created_at,
            expires_at=k.expires_at,
            is_active=k.is_active,
        )
        for k in auth.api_keys.values()
    ]


@router.post(
    "/api-keys",
    response_model=APIKeyResponse,
    summary="Create API key",
    description="Create a new API key. **The key is only shown once!**",
    dependencies=[Depends(require_operator())],
)
async def create_api_key(request: CreateAPIKeyRequest):
    """
    Create a new API key for device authentication.

    **Important:** The key value is only returned once. Store it securely!
    """
    auth = get_auth_manager()

    expires_in = None
    if request.expires_in_days:
        expires_in = timedelta(days=request.expires_in_days)

    key, api_key = auth.create_api_key(
        name=request.name,
        device_id=request.device_id,
        expires_in=expires_in,
    )

    return APIKeyResponse(
        key_id=api_key.key_id,
        key=key,  # Only time the actual key is returned!
        name=api_key.name,
        device_id=api_key.device_id,
        role=api_key.role.value,
        created_at=api_key.created_at,
        expires_at=api_key.expires_at,
        is_active=api_key.is_active,
    )


@router.delete(
    "/api-keys/{key_id}",
    summary="Revoke API key",
    description="Revoke an API key.",
    dependencies=[Depends(require_operator())],
)
async def revoke_api_key(key_id: str):
    """Revoke an API key."""
    auth = get_auth_manager()

    if key_id not in auth.api_keys:
        raise HTTPException(status_code=404, detail="API key not found")

    auth.revoke_api_key(key_id)
    return {"message": "API key revoked"}
