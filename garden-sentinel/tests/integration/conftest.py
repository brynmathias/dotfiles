"""
Shared fixtures for integration tests.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
import sqlite3


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    # Create basic schema
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'viewer',
            created_at REAL NOT NULL
        );

        CREATE TABLE api_keys (
            id TEXT PRIMARY KEY,
            key_hash TEXT NOT NULL,
            name TEXT NOT NULL,
            device_id TEXT,
            created_at REAL NOT NULL,
            last_used REAL
        );

        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at REAL NOT NULL,
            expires_at REAL NOT NULL,
            revoked INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE detections (
            id TEXT PRIMARY KEY,
            device_id TEXT NOT NULL,
            timestamp REAL NOT NULL,
            predator_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            bbox TEXT,
            frame_path TEXT,
            processed INTEGER DEFAULT 0
        );

        CREATE TABLE alerts (
            id TEXT PRIMARY KEY,
            detection_id TEXT,
            device_id TEXT NOT NULL,
            timestamp REAL NOT NULL,
            severity TEXT NOT NULL,
            predator_type TEXT,
            confidence REAL,
            acknowledged INTEGER DEFAULT 0,
            acknowledged_by TEXT,
            acknowledged_at REAL,
            FOREIGN KEY (detection_id) REFERENCES detections(id)
        );

        CREATE TABLE deterrence_actions (
            id TEXT PRIMARY KEY,
            alert_id TEXT,
            device_id TEXT NOT NULL,
            timestamp REAL NOT NULL,
            action_type TEXT NOT NULL,
            parameters TEXT,
            success INTEGER,
            FOREIGN KEY (alert_id) REFERENCES alerts(id)
        );

        CREATE TABLE devices (
            id TEXT PRIMARY KEY,
            name TEXT,
            type TEXT NOT NULL,
            last_seen REAL,
            status TEXT DEFAULT 'unknown',
            config TEXT
        );

        CREATE TABLE zones (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            geometry TEXT NOT NULL,
            zone_type TEXT NOT NULL,
            properties TEXT
        );

        CREATE TABLE settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE INDEX idx_detections_timestamp ON detections(timestamp);
        CREATE INDEX idx_alerts_timestamp ON alerts(timestamp);
        CREATE INDEX idx_alerts_acknowledged ON alerts(acknowledged);
    """)
    conn.close()

    yield db_path

    # Cleanup
    os.unlink(db_path)


@pytest.fixture
def temp_storage():
    """Create a temporary storage directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir)
        (storage_path / "frames").mkdir()
        (storage_path / "recordings").mkdir()
        (storage_path / "sounds").mkdir()
        yield storage_path


@pytest.fixture
def mock_camera():
    """Create a mock camera object."""
    camera = MagicMock()
    camera.device_id = "test-camera-01"
    camera.is_running = True
    camera.fps = 25.0
    camera.resolution = (1920, 1080)
    camera.get_frame = MagicMock(return_value=b"mock_frame_data")
    return camera


@pytest.fixture
def mock_detector():
    """Create a mock detection model."""
    detector = MagicMock()
    detector.detect = MagicMock(return_value=[
        {
            "class": "fox",
            "confidence": 0.92,
            "bbox": [100, 100, 200, 200],
        }
    ])
    return detector


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket connection."""
    ws = AsyncMock()
    ws.send_json = AsyncMock()
    ws.receive_json = AsyncMock(return_value={"type": "ping"})
    ws.close = AsyncMock()
    return ws


@pytest.fixture
def sample_detection():
    """Sample detection event data."""
    return {
        "id": "det-001",
        "device_id": "camera-north-01",
        "timestamp": 1700000000.0,
        "predator_type": "fox",
        "confidence": 0.92,
        "bbox": [100, 100, 200, 200],
    }


@pytest.fixture
def sample_alert():
    """Sample alert data."""
    return {
        "id": "alert-001",
        "detection_id": "det-001",
        "device_id": "camera-north-01",
        "timestamp": 1700000000.0,
        "severity": "high",
        "predator_type": "fox",
        "confidence": 0.92,
        "acknowledged": False,
    }


@pytest.fixture
def sample_user():
    """Sample user data."""
    return {
        "id": "user-001",
        "username": "testuser",
        "password": "testpass123",
        "role": "admin",
    }


@pytest.fixture
def sample_device():
    """Sample device registration data."""
    return {
        "id": "camera-north-01",
        "name": "North Garden Camera",
        "type": "camera",
        "status": "online",
        "config": {
            "resolution": [1920, 1080],
            "fps": 25,
            "deterrents": ["spray", "sound"],
        },
    }


@pytest.fixture
def sample_zone():
    """Sample zone data."""
    return {
        "id": "zone-001",
        "name": "Chicken Coop",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
        },
        "zone_type": "protected",
        "properties": {
            "priority": "high",
            "deterrence_level": "aggressive",
        },
    }
