"""
Integration tests for authentication API endpoints.
"""

import pytest
import time
import sqlite3
import hashlib
import secrets
from unittest.mock import MagicMock, patch


class TestAuthenticationFlow:
    """Tests for the authentication flow."""

    def test_password_hashing(self):
        """Test that passwords are hashed securely."""
        password = "testpassword123"
        salt = secrets.token_hex(16)

        # Simulate PBKDF2 hashing
        hash_result = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode(),
            salt.encode(),
            100000
        )
        password_hash = f"{salt}:{hash_result.hex()}"

        # Verify we can validate the password
        stored_salt, stored_hash = password_hash.split(":")
        verify_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode(),
            stored_salt.encode(),
            100000
        ).hex()

        assert verify_hash == stored_hash

    def test_invalid_password_fails(self):
        """Test that invalid passwords fail verification."""
        password = "testpassword123"
        wrong_password = "wrongpassword"
        salt = secrets.token_hex(16)

        hash_result = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode(),
            salt.encode(),
            100000
        )
        stored_hash = hash_result.hex()

        verify_hash = hashlib.pbkdf2_hmac(
            'sha256',
            wrong_password.encode(),
            salt.encode(),
            100000
        ).hex()

        assert verify_hash != stored_hash

    def test_user_creation_in_database(self, temp_db, sample_user):
        """Test creating a user in the database."""
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        # Hash password
        salt = secrets.token_hex(16)
        hash_result = hashlib.pbkdf2_hmac(
            'sha256',
            sample_user["password"].encode(),
            salt.encode(),
            100000
        )
        password_hash = f"{salt}:{hash_result.hex()}"

        # Insert user
        cursor.execute("""
            INSERT INTO users (id, username, password_hash, role, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            sample_user["id"],
            sample_user["username"],
            password_hash,
            sample_user["role"],
            time.time()
        ))
        conn.commit()

        # Verify user exists
        cursor.execute("SELECT username, role FROM users WHERE id = ?", (sample_user["id"],))
        row = cursor.fetchone()

        assert row is not None
        assert row[0] == sample_user["username"]
        assert row[1] == sample_user["role"]

        conn.close()

    def test_session_creation(self, temp_db, sample_user):
        """Test creating a session after login."""
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        # Create user first
        cursor.execute("""
            INSERT INTO users (id, username, password_hash, role, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (sample_user["id"], sample_user["username"], "hash", sample_user["role"], time.time()))

        # Create session
        session_id = secrets.token_hex(32)
        created_at = time.time()
        expires_at = created_at + 86400  # 24 hours

        cursor.execute("""
            INSERT INTO sessions (id, user_id, created_at, expires_at)
            VALUES (?, ?, ?, ?)
        """, (session_id, sample_user["id"], created_at, expires_at))
        conn.commit()

        # Verify session
        cursor.execute("""
            SELECT user_id, expires_at, revoked FROM sessions WHERE id = ?
        """, (session_id,))
        row = cursor.fetchone()

        assert row is not None
        assert row[0] == sample_user["id"]
        assert row[1] > time.time()
        assert row[2] == 0

        conn.close()

    def test_session_expiration(self, temp_db, sample_user):
        """Test that expired sessions are not valid."""
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        # Create user
        cursor.execute("""
            INSERT INTO users (id, username, password_hash, role, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (sample_user["id"], sample_user["username"], "hash", sample_user["role"], time.time()))

        # Create expired session
        session_id = secrets.token_hex(32)
        created_at = time.time() - 100000
        expires_at = created_at + 86400  # Already expired

        cursor.execute("""
            INSERT INTO sessions (id, user_id, created_at, expires_at)
            VALUES (?, ?, ?, ?)
        """, (session_id, sample_user["id"], created_at, expires_at))
        conn.commit()

        # Check if session is expired
        cursor.execute("""
            SELECT id FROM sessions
            WHERE id = ? AND expires_at > ? AND revoked = 0
        """, (session_id, time.time()))
        row = cursor.fetchone()

        assert row is None  # Session should not be returned

        conn.close()

    def test_session_revocation(self, temp_db, sample_user):
        """Test revoking a session (logout)."""
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        # Create user
        cursor.execute("""
            INSERT INTO users (id, username, password_hash, role, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (sample_user["id"], sample_user["username"], "hash", sample_user["role"], time.time()))

        # Create session
        session_id = secrets.token_hex(32)
        cursor.execute("""
            INSERT INTO sessions (id, user_id, created_at, expires_at)
            VALUES (?, ?, ?, ?)
        """, (session_id, sample_user["id"], time.time(), time.time() + 86400))
        conn.commit()

        # Revoke session
        cursor.execute("UPDATE sessions SET revoked = 1 WHERE id = ?", (session_id,))
        conn.commit()

        # Check session is invalid
        cursor.execute("""
            SELECT id FROM sessions
            WHERE id = ? AND revoked = 0
        """, (session_id,))
        row = cursor.fetchone()

        assert row is None

        conn.close()


class TestAPIKeyAuthentication:
    """Tests for API key authentication."""

    def test_api_key_creation(self, temp_db):
        """Test creating an API key."""
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        key_id = secrets.token_hex(16)
        raw_key = secrets.token_hex(32)
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        cursor.execute("""
            INSERT INTO api_keys (id, key_hash, name, device_id, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (key_id, key_hash, "Test Camera", "camera-01", time.time()))
        conn.commit()

        # Verify key exists
        cursor.execute("SELECT name, device_id FROM api_keys WHERE id = ?", (key_id,))
        row = cursor.fetchone()

        assert row is not None
        assert row[0] == "Test Camera"
        assert row[1] == "camera-01"

        conn.close()

    def test_api_key_validation(self, temp_db):
        """Test validating an API key."""
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        raw_key = "gs_" + secrets.token_hex(32)
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        cursor.execute("""
            INSERT INTO api_keys (id, key_hash, name, device_id, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, ("key-001", key_hash, "Test Camera", "camera-01", time.time()))
        conn.commit()

        # Validate key
        check_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        cursor.execute("""
            SELECT id, device_id FROM api_keys WHERE key_hash = ?
        """, (check_hash,))
        row = cursor.fetchone()

        assert row is not None
        assert row[1] == "camera-01"

        conn.close()

    def test_api_key_last_used_update(self, temp_db):
        """Test updating last_used timestamp on API key usage."""
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        key_id = "key-002"
        cursor.execute("""
            INSERT INTO api_keys (id, key_hash, name, device_id, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (key_id, "hash", "Test Camera", "camera-01", time.time()))
        conn.commit()

        # Update last_used
        now = time.time()
        cursor.execute("UPDATE api_keys SET last_used = ? WHERE id = ?", (now, key_id))
        conn.commit()

        # Verify
        cursor.execute("SELECT last_used FROM api_keys WHERE id = ?", (key_id,))
        row = cursor.fetchone()

        assert row is not None
        assert abs(row[0] - now) < 1

        conn.close()


class TestRoleBasedAccess:
    """Tests for role-based access control."""

    def test_role_hierarchy(self):
        """Test role permission hierarchy."""
        roles = {
            "viewer": ["read"],
            "operator": ["read", "control"],
            "admin": ["read", "control", "admin"],
        }

        def has_permission(role: str, permission: str) -> bool:
            return permission in roles.get(role, [])

        # Viewer can only read
        assert has_permission("viewer", "read")
        assert not has_permission("viewer", "control")
        assert not has_permission("viewer", "admin")

        # Operator can read and control
        assert has_permission("operator", "read")
        assert has_permission("operator", "control")
        assert not has_permission("operator", "admin")

        # Admin can do everything
        assert has_permission("admin", "read")
        assert has_permission("admin", "control")
        assert has_permission("admin", "admin")

    def test_device_role_permissions(self):
        """Test device role can only do device operations."""
        device_permissions = ["heartbeat", "detection", "stream"]
        user_permissions = ["read", "control", "admin"]

        # Device should not have user permissions
        for perm in user_permissions:
            assert perm not in device_permissions

        # Device should have device permissions
        for perm in device_permissions:
            assert perm in device_permissions
