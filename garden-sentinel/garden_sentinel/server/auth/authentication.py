"""
Authentication and authorization for Garden Sentinel.

Provides:
- JWT-based authentication
- API key authentication for edge devices
- User management with roles
- Session management
"""

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
import base64
import json


class UserRole(Enum):
    """User roles with different permission levels."""
    VIEWER = "viewer"        # Can view dashboards and alerts
    OPERATOR = "operator"    # Can control cameras and deterrents
    ADMIN = "admin"          # Full access including user management
    DEVICE = "device"        # Edge device API access


@dataclass
class User:
    """User account."""
    user_id: str
    username: str
    email: str
    password_hash: str
    salt: str
    role: UserRole
    created_at: float = field(default_factory=time.time)
    last_login: Optional[float] = None
    is_active: bool = True
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None

    def check_password(self, password: str) -> bool:
        """Verify password against stored hash."""
        computed_hash = hash_password(password, self.salt)
        return hmac.compare_digest(computed_hash, self.password_hash)

    def to_dict(self) -> dict:
        """Convert to dictionary (excluding sensitive fields)."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "role": self.role.value,
            "created_at": self.created_at,
            "last_login": self.last_login,
            "is_active": self.is_active,
            "mfa_enabled": self.mfa_enabled,
        }


@dataclass
class APIKey:
    """API key for device or service authentication."""
    key_id: str
    key_hash: str  # Store hash, not the actual key
    name: str
    device_id: Optional[str] = None  # Associated device if applicable
    role: UserRole = UserRole.DEVICE
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    last_used: Optional[float] = None
    is_active: bool = True

    def is_expired(self) -> bool:
        """Check if the key has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


@dataclass
class Session:
    """User session."""
    session_id: str
    user_id: str
    created_at: float
    expires_at: float
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_valid: bool = True

    def is_expired(self) -> bool:
        """Check if the session has expired."""
        return time.time() > self.expires_at


def generate_salt() -> str:
    """Generate a random salt for password hashing."""
    return secrets.token_hex(32)


def hash_password(password: str, salt: str) -> str:
    """Hash a password with salt using PBKDF2."""
    return hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000  # iterations
    ).hex()


def generate_api_key() -> tuple[str, str]:
    """
    Generate a new API key.

    Returns tuple of (key, key_hash).
    The key should be shown to user once, only the hash is stored.
    """
    key = f"gs_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    return key, key_hash


def verify_api_key(key: str, key_hash: str) -> bool:
    """Verify an API key against its hash."""
    computed_hash = hashlib.sha256(key.encode()).hexdigest()
    return hmac.compare_digest(computed_hash, key_hash)


class JWTManager:
    """
    Simple JWT implementation for authentication tokens.

    Note: For production, consider using a library like PyJWT.
    This is a minimal implementation for the project.
    """

    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm

    def create_token(
        self,
        payload: dict,
        expires_in: timedelta = timedelta(hours=24),
    ) -> str:
        """Create a JWT token."""
        now = time.time()

        token_payload = {
            **payload,
            "iat": now,
            "exp": now + expires_in.total_seconds(),
        }

        # Create header
        header = {"alg": self.algorithm, "typ": "JWT"}

        # Encode
        header_b64 = self._base64url_encode(json.dumps(header))
        payload_b64 = self._base64url_encode(json.dumps(token_payload))

        # Sign
        signature = self._sign(f"{header_b64}.{payload_b64}")
        signature_b64 = self._base64url_encode(signature)

        return f"{header_b64}.{payload_b64}.{signature_b64}"

    def verify_token(self, token: str) -> Optional[dict]:
        """
        Verify and decode a JWT token.

        Returns payload if valid, None if invalid or expired.
        """
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None

            header_b64, payload_b64, signature_b64 = parts

            # Verify signature
            expected_signature = self._sign(f"{header_b64}.{payload_b64}")
            expected_signature_b64 = self._base64url_encode(expected_signature)

            if not hmac.compare_digest(signature_b64, expected_signature_b64):
                return None

            # Decode payload
            payload = json.loads(self._base64url_decode(payload_b64))

            # Check expiration
            if payload.get("exp", 0) < time.time():
                return None

            return payload

        except Exception:
            return None

    def _sign(self, message: str) -> bytes:
        """Create HMAC signature."""
        return hmac.new(
            self.secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).digest()

    def _base64url_encode(self, data) -> str:
        """Base64URL encode."""
        if isinstance(data, str):
            data = data.encode()
        return base64.urlsafe_b64encode(data).rstrip(b'=').decode()

    def _base64url_decode(self, data: str) -> bytes:
        """Base64URL decode."""
        padding = 4 - len(data) % 4
        if padding != 4:
            data += '=' * padding
        return base64.urlsafe_b64decode(data)


class AuthManager:
    """
    Manages authentication for Garden Sentinel.

    Handles user authentication, API keys, and sessions.
    """

    def __init__(
        self,
        secret_key: str,
        token_expiry: timedelta = timedelta(hours=24),
        session_expiry: timedelta = timedelta(days=7),
    ):
        self.jwt = JWTManager(secret_key)
        self.token_expiry = token_expiry
        self.session_expiry = session_expiry

        # In-memory storage (replace with database in production)
        self.users: dict[str, User] = {}
        self.api_keys: dict[str, APIKey] = {}
        self.sessions: dict[str, Session] = {}

        # Create default admin if no users exist
        self._ensure_default_admin()

    def _ensure_default_admin(self):
        """Create default admin user if none exists."""
        if not self.users:
            self.create_user(
                username="admin",
                email="admin@localhost",
                password="changeme",  # User must change this!
                role=UserRole.ADMIN,
            )

    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        role: UserRole = UserRole.VIEWER,
    ) -> User:
        """Create a new user."""
        user_id = secrets.token_urlsafe(16)
        salt = generate_salt()
        password_hash = hash_password(password, salt)

        user = User(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            salt=salt,
            role=role,
        )

        self.users[user_id] = user
        return user

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user by username and password."""
        for user in self.users.values():
            if user.username == username and user.is_active:
                if user.check_password(password):
                    user.last_login = time.time()
                    return user
        return None

    def create_session(
        self,
        user: User,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> tuple[str, Session]:
        """Create a new session for a user. Returns (token, session)."""
        session_id = secrets.token_urlsafe(32)
        now = time.time()

        session = Session(
            session_id=session_id,
            user_id=user.user_id,
            created_at=now,
            expires_at=now + self.session_expiry.total_seconds(),
            ip_address=ip_address,
            user_agent=user_agent,
        )

        self.sessions[session_id] = session

        # Create JWT token
        token = self.jwt.create_token(
            payload={
                "sub": user.user_id,
                "sid": session_id,
                "role": user.role.value,
                "username": user.username,
            },
            expires_in=self.token_expiry,
        )

        return token, session

    def verify_token(self, token: str) -> Optional[dict]:
        """Verify a JWT token and return its payload."""
        payload = self.jwt.verify_token(token)
        if not payload:
            return None

        # Check session is still valid
        session_id = payload.get("sid")
        if session_id:
            session = self.sessions.get(session_id)
            if not session or not session.is_valid or session.is_expired():
                return None

        return payload

    def get_user_from_token(self, token: str) -> Optional[User]:
        """Get user from a valid token."""
        payload = self.verify_token(token)
        if not payload:
            return None

        user_id = payload.get("sub")
        return self.users.get(user_id)

    def invalidate_session(self, session_id: str):
        """Invalidate a session (logout)."""
        if session_id in self.sessions:
            self.sessions[session_id].is_valid = False

    def create_api_key(
        self,
        name: str,
        device_id: Optional[str] = None,
        role: UserRole = UserRole.DEVICE,
        expires_in: Optional[timedelta] = None,
    ) -> tuple[str, APIKey]:
        """
        Create a new API key.

        Returns (key, api_key_object). The key is only returned once!
        """
        key, key_hash = generate_api_key()
        key_id = secrets.token_urlsafe(16)

        expires_at = None
        if expires_in:
            expires_at = time.time() + expires_in.total_seconds()

        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            device_id=device_id,
            role=role,
            expires_at=expires_at,
        )

        self.api_keys[key_id] = api_key

        return key, api_key

    def verify_api_key(self, key: str) -> Optional[APIKey]:
        """Verify an API key and return its details if valid."""
        for api_key in self.api_keys.values():
            if verify_api_key(key, api_key.key_hash):
                if api_key.is_active and not api_key.is_expired():
                    api_key.last_used = time.time()
                    return api_key
        return None

    def revoke_api_key(self, key_id: str):
        """Revoke an API key."""
        if key_id in self.api_keys:
            self.api_keys[key_id].is_active = False

    def change_password(self, user_id: str, new_password: str) -> bool:
        """Change a user's password."""
        user = self.users.get(user_id)
        if not user:
            return False

        user.salt = generate_salt()
        user.password_hash = hash_password(new_password, user.salt)
        return True

    def has_permission(self, user_or_key, required_role: UserRole) -> bool:
        """Check if user or API key has required permission level."""
        role_hierarchy = {
            UserRole.VIEWER: 0,
            UserRole.DEVICE: 1,
            UserRole.OPERATOR: 2,
            UserRole.ADMIN: 3,
        }

        if isinstance(user_or_key, User):
            user_level = role_hierarchy.get(user_or_key.role, 0)
        elif isinstance(user_or_key, APIKey):
            user_level = role_hierarchy.get(user_or_key.role, 0)
        else:
            return False

        required_level = role_hierarchy.get(required_role, 0)
        return user_level >= required_level
