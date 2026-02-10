"""
FastAPI authentication middleware and dependencies.

Provides:
- JWT token validation
- API key validation
- Role-based access control
- Rate limiting
"""

import time
from collections import defaultdict
from functools import wraps
from typing import Optional, Callable

from fastapi import Request, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader

from .authentication import AuthManager, User, APIKey, UserRole


# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
    ):
        self.rpm = requests_per_minute
        self.rph = requests_per_hour
        self.minute_counts: dict[str, list[float]] = defaultdict(list)
        self.hour_counts: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed."""
        now = time.time()
        minute_ago = now - 60
        hour_ago = now - 3600

        # Clean old entries
        self.minute_counts[identifier] = [
            t for t in self.minute_counts[identifier] if t > minute_ago
        ]
        self.hour_counts[identifier] = [
            t for t in self.hour_counts[identifier] if t > hour_ago
        ]

        # Check limits
        if len(self.minute_counts[identifier]) >= self.rpm:
            return False
        if len(self.hour_counts[identifier]) >= self.rph:
            return False

        # Record request
        self.minute_counts[identifier].append(now)
        self.hour_counts[identifier].append(now)

        return True

    def get_remaining(self, identifier: str) -> dict:
        """Get remaining requests."""
        now = time.time()
        minute_ago = now - 60
        hour_ago = now - 3600

        minute_count = len([
            t for t in self.minute_counts[identifier] if t > minute_ago
        ])
        hour_count = len([
            t for t in self.hour_counts[identifier] if t > hour_ago
        ])

        return {
            "minute": self.rpm - minute_count,
            "hour": self.rph - hour_count,
        }


# Global instances (will be initialized by the app)
_auth_manager: Optional[AuthManager] = None
_rate_limiter: Optional[RateLimiter] = None


def init_auth(auth_manager: AuthManager, rate_limiter: Optional[RateLimiter] = None):
    """Initialize authentication globals."""
    global _auth_manager, _rate_limiter
    _auth_manager = auth_manager
    _rate_limiter = rate_limiter or RateLimiter()


def get_auth_manager() -> AuthManager:
    """Get the auth manager instance."""
    if _auth_manager is None:
        raise RuntimeError("Auth manager not initialized")
    return _auth_manager


async def get_client_ip(request: Request) -> str:
    """Extract client IP from request."""
    # Check for proxy headers
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def rate_limit(request: Request):
    """Rate limiting dependency."""
    if _rate_limiter is None:
        return

    client_ip = await get_client_ip(request)

    if not _rate_limiter.is_allowed(client_ip):
        remaining = _rate_limiter.get_remaining(client_ip)
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={
                "X-RateLimit-Remaining-Minute": str(remaining["minute"]),
                "X-RateLimit-Remaining-Hour": str(remaining["hour"]),
                "Retry-After": "60",
            }
        )


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    api_key: Optional[str] = Depends(api_key_header),
) -> Optional[User | APIKey]:
    """
    Get current user from token or API key (optional).

    Returns None if no valid authentication provided.
    """
    auth = get_auth_manager()

    # Try JWT token first
    if credentials and credentials.credentials:
        user = auth.get_user_from_token(credentials.credentials)
        if user:
            return user

    # Try API key
    if api_key:
        key = auth.verify_api_key(api_key)
        if key:
            return key

    return None


async def get_current_user(
    auth_result: Optional[User | APIKey] = Depends(get_current_user_optional),
) -> User | APIKey:
    """
    Get current user from token or API key (required).

    Raises 401 if no valid authentication.
    """
    if auth_result is None:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return auth_result


def require_role(required_role: UserRole):
    """
    Dependency that requires a specific role.

    Usage:
        @app.get("/admin", dependencies=[Depends(require_role(UserRole.ADMIN))])
    """
    async def check_role(
        current_user: User | APIKey = Depends(get_current_user),
    ):
        auth = get_auth_manager()
        if not auth.has_permission(current_user, required_role):
            raise HTTPException(
                status_code=403,
                detail=f"Required role: {required_role.value}",
            )
        return current_user

    return check_role


def require_viewer():
    """Require at least viewer role."""
    return require_role(UserRole.VIEWER)


def require_operator():
    """Require at least operator role."""
    return require_role(UserRole.OPERATOR)


def require_admin():
    """Require admin role."""
    return require_role(UserRole.ADMIN)


def require_device():
    """Require device or higher role."""
    return require_role(UserRole.DEVICE)


class AuthenticatedUser:
    """
    Context for the current authenticated user.

    Can be used as a dependency to get structured auth info.
    """

    def __init__(
        self,
        user: Optional[User] = None,
        api_key: Optional[APIKey] = None,
    ):
        self.user = user
        self.api_key = api_key

    @property
    def is_authenticated(self) -> bool:
        return self.user is not None or self.api_key is not None

    @property
    def is_user(self) -> bool:
        return self.user is not None

    @property
    def is_device(self) -> bool:
        return self.api_key is not None

    @property
    def role(self) -> Optional[UserRole]:
        if self.user:
            return self.user.role
        if self.api_key:
            return self.api_key.role
        return None

    @property
    def identifier(self) -> Optional[str]:
        """Get user ID or device ID."""
        if self.user:
            return self.user.user_id
        if self.api_key:
            return self.api_key.device_id or self.api_key.key_id
        return None


async def get_authenticated_user(
    auth_result: Optional[User | APIKey] = Depends(get_current_user_optional),
) -> AuthenticatedUser:
    """Get structured authentication context."""
    if isinstance(auth_result, User):
        return AuthenticatedUser(user=auth_result)
    elif isinstance(auth_result, APIKey):
        return AuthenticatedUser(api_key=auth_result)
    return AuthenticatedUser()


# Middleware for adding auth headers to responses
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)

    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"

    return response
