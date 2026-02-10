# Authentication and security module
from .authentication import (
    AuthManager,
    User,
    UserRole,
    APIKey,
    Session,
    JWTManager,
    hash_password,
    generate_api_key,
    verify_api_key,
)
from .middleware import (
    init_auth,
    get_auth_manager,
    get_current_user,
    get_current_user_optional,
    get_authenticated_user,
    AuthenticatedUser,
    require_role,
    require_viewer,
    require_operator,
    require_admin,
    require_device,
    rate_limit,
    RateLimiter,
    add_security_headers,
)
from .tls import (
    TLSConfig,
    create_ssl_context,
    create_client_ssl_context,
    generate_self_signed_cert,
    get_cert_info,
)

__all__ = [
    # Authentication
    "AuthManager",
    "User",
    "UserRole",
    "APIKey",
    "Session",
    "JWTManager",
    "hash_password",
    "generate_api_key",
    "verify_api_key",
    # Middleware
    "init_auth",
    "get_auth_manager",
    "get_current_user",
    "get_current_user_optional",
    "get_authenticated_user",
    "AuthenticatedUser",
    "require_role",
    "require_viewer",
    "require_operator",
    "require_admin",
    "require_device",
    "rate_limit",
    "RateLimiter",
    "add_security_headers",
    # TLS
    "TLSConfig",
    "create_ssl_context",
    "create_client_ssl_context",
    "generate_self_signed_cert",
    "get_cert_info",
]
