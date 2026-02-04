"""
Storage manager for frames, events, and training data.
"""

import json
import logging
import os
import shutil
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

from garden_sentinel.shared import AlertEvent, Detection


@dataclass
class StorageConfig:
    frames_dir: str = "data/frames"
    events_dir: str = "data/events"
    training_data_dir: str = "data/training"
    database_url: str = "sqlite:///data/garden_sentinel.db"
    max_frames: int = 10000
    frame_retention_days: int = 7
    event_retention_days: int = 30
    auto_collect_detections: bool = True


class StorageManager:
    """
    Manages storage of frames, events, and training data.
    """

    def __init__(self, config: StorageConfig):
        self.config = config
        self._db_conn: Optional[sqlite3.Connection] = None
        self._db_lock = threading.Lock()

        # Create directories
        for dir_path in [config.frames_dir, config.events_dir, config.training_data_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database."""
        db_path = self.config.database_url.replace("sqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._db_conn = sqlite3.connect(db_path, check_same_thread=False)

        # Create tables
        with self._db_lock:
            cursor = self._db_conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    device_id TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    threat_level TEXT NOT NULL,
                    detections_json TEXT NOT NULL,
                    frame_path TEXT,
                    actions_taken TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS frames (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    file_path TEXT NOT NULL,
                    has_detections BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT NOT NULL,
                    annotation_path TEXT,
                    class_name TEXT NOT NULL,
                    source_device TEXT,
                    source_event TEXT,
                    verified BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS devices (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    location TEXT,
                    last_seen DATETIME,
                    status TEXT DEFAULT 'unknown'
                )
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_device ON events(device_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_frames_device ON frames(device_id)")

            self._db_conn.commit()

        logger.info("Database initialized")

    def save_frame(
        self,
        device_id: str,
        frame: np.ndarray,
        timestamp: float,
        has_detections: bool = False,
    ) -> str:
        """
        Save a frame to storage.

        Returns:
            Path to saved frame
        """
        # Create device subdirectory
        device_dir = Path(self.config.frames_dir) / device_id
        device_dir.mkdir(exist_ok=True)

        # Generate filename
        dt = datetime.fromtimestamp(timestamp)
        filename = f"{dt.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        file_path = device_dir / filename

        # Save image
        cv2.imwrite(str(file_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

        # Record in database
        with self._db_lock:
            cursor = self._db_conn.cursor()
            cursor.execute(
                """
                INSERT INTO frames (device_id, timestamp, file_path, has_detections)
                VALUES (?, ?, ?, ?)
                """,
                (device_id, datetime.fromtimestamp(timestamp), str(file_path), has_detections),
            )
            self._db_conn.commit()

        return str(file_path)

    def save_event(self, event: AlertEvent, frame: Optional[np.ndarray] = None) -> str:
        """
        Save an alert event.

        Returns:
            Path to saved event data
        """
        # Create event directory
        event_dir = Path(self.config.events_dir) / event.event_id
        event_dir.mkdir(parents=True, exist_ok=True)

        # Save frame if provided
        frame_path = None
        if frame is not None:
            frame_path = event_dir / "frame.jpg"
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Save event JSON
        event_data = event.to_dict()
        event_data["frame_path"] = str(frame_path) if frame_path else None

        event_file = event_dir / "event.json"
        with open(event_file, "w") as f:
            json.dump(event_data, f, indent=2, default=str)

        # Record in database
        with self._db_lock:
            cursor = self._db_conn.cursor()
            cursor.execute(
                """
                INSERT INTO events (id, device_id, timestamp, threat_level, detections_json, frame_path, actions_taken)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.device_id,
                    event.timestamp,
                    event.max_threat_level.value,
                    json.dumps([d.to_dict() for d in event.detections]),
                    str(frame_path) if frame_path else None,
                    json.dumps(event.actions_taken),
                ),
            )
            self._db_conn.commit()

        # Auto-collect for training
        if self.config.auto_collect_detections and frame is not None:
            self._collect_training_samples(event, frame)

        return str(event_dir)

    def _collect_training_samples(self, event: AlertEvent, frame: np.ndarray):
        """Collect detection samples for training data."""
        height, width = frame.shape[:2]

        for det in event.detections:
            # Extract crop
            x1 = int(det.bbox.x * width)
            y1 = int(det.bbox.y * height)
            x2 = int((det.bbox.x + det.bbox.width) * width)
            y2 = int((det.bbox.y + det.bbox.height) * height)

            # Add padding
            pad = 20
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(width, x2 + pad)
            y2 = min(height, y2 + pad)

            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # Save crop
            class_dir = Path(self.config.training_data_dir) / "images" / det.class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            image_path = class_dir / f"{timestamp}.jpg"
            cv2.imwrite(str(image_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Create YOLO annotation
            ann_dir = Path(self.config.training_data_dir) / "labels" / det.class_name
            ann_dir.mkdir(parents=True, exist_ok=True)

            # YOLO format: class_id x_center y_center width height (normalized)
            # For crops, the detection fills most of the image
            ann_path = ann_dir / f"{timestamp}.txt"
            with open(ann_path, "w") as f:
                f.write(f"0 0.5 0.5 0.9 0.9\n")  # Simplified annotation for crop

            # Record in database
            with self._db_lock:
                cursor = self._db_conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO training_samples (image_path, annotation_path, class_name, source_device, source_event)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (str(image_path), str(ann_path), det.class_name, event.device_id, event.event_id),
                )
                self._db_conn.commit()

    def get_events(
        self,
        device_id: Optional[str] = None,
        threat_level: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query events from database."""
        query = "SELECT * FROM events WHERE 1=1"
        params = []

        if device_id:
            query += " AND device_id = ?"
            params.append(device_id)

        if threat_level:
            query += " AND threat_level = ?"
            params.append(threat_level)

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._db_lock:
            cursor = self._db_conn.cursor()
            cursor.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

        return [dict(zip(columns, row)) for row in rows]

    def get_training_samples(self, class_name: Optional[str] = None, verified_only: bool = False) -> list[dict]:
        """Get training samples."""
        query = "SELECT * FROM training_samples WHERE 1=1"
        params = []

        if class_name:
            query += " AND class_name = ?"
            params.append(class_name)

        if verified_only:
            query += " AND verified = 1"

        with self._db_lock:
            cursor = self._db_conn.cursor()
            cursor.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

        return [dict(zip(columns, row)) for row in rows]

    def mark_sample_verified(self, sample_id: int, verified: bool = True):
        """Mark a training sample as verified."""
        with self._db_lock:
            cursor = self._db_conn.cursor()
            cursor.execute(
                "UPDATE training_samples SET verified = ? WHERE id = ?",
                (verified, sample_id),
            )
            self._db_conn.commit()

    def update_device(self, device_id: str, name: Optional[str] = None, location: Optional[str] = None):
        """Update or insert device information."""
        with self._db_lock:
            cursor = self._db_conn.cursor()
            cursor.execute(
                """
                INSERT INTO devices (id, name, location, last_seen, status)
                VALUES (?, ?, ?, ?, 'online')
                ON CONFLICT(id) DO UPDATE SET
                    name = COALESCE(?, name),
                    location = COALESCE(?, location),
                    last_seen = ?,
                    status = 'online'
                """,
                (device_id, name, location, datetime.now(), name, location, datetime.now()),
            )
            self._db_conn.commit()

    def get_devices(self) -> list[dict]:
        """Get all registered devices."""
        with self._db_lock:
            cursor = self._db_conn.cursor()
            cursor.execute("SELECT * FROM devices")
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

        return [dict(zip(columns, row)) for row in rows]

    def cleanup_old_data(self):
        """Remove old frames and events based on retention policy."""
        # Clean up old frames
        frame_cutoff = datetime.now() - timedelta(days=self.config.frame_retention_days)

        with self._db_lock:
            cursor = self._db_conn.cursor()
            cursor.execute(
                "SELECT file_path FROM frames WHERE timestamp < ?",
                (frame_cutoff,),
            )
            old_frames = cursor.fetchall()

            for (file_path,) in old_frames:
                try:
                    Path(file_path).unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Failed to delete frame {file_path}: {e}")

            cursor.execute("DELETE FROM frames WHERE timestamp < ?", (frame_cutoff,))
            deleted_frames = cursor.rowcount

        # Clean up old events
        event_cutoff = datetime.now() - timedelta(days=self.config.event_retention_days)

        with self._db_lock:
            cursor = self._db_conn.cursor()
            cursor.execute(
                "SELECT id FROM events WHERE timestamp < ?",
                (event_cutoff,),
            )
            old_events = cursor.fetchall()

            for (event_id,) in old_events:
                event_dir = Path(self.config.events_dir) / event_id
                if event_dir.exists():
                    shutil.rmtree(event_dir, ignore_errors=True)

            cursor.execute("DELETE FROM events WHERE timestamp < ?", (event_cutoff,))
            deleted_events = cursor.rowcount

            self._db_conn.commit()

        logger.info(f"Cleanup: removed {deleted_frames} frames, {deleted_events} events")

    def get_storage_stats(self) -> dict:
        """Get storage statistics."""
        stats = {}

        # Count records
        with self._db_lock:
            cursor = self._db_conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM frames")
            stats["total_frames"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM events")
            stats["total_events"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM training_samples")
            stats["total_training_samples"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM training_samples WHERE verified = 1")
            stats["verified_training_samples"] = cursor.fetchone()[0]

        # Calculate disk usage
        for name, path in [
            ("frames", self.config.frames_dir),
            ("events", self.config.events_dir),
            ("training", self.config.training_data_dir),
        ]:
            total_size = sum(
                f.stat().st_size for f in Path(path).rglob("*") if f.is_file()
            )
            stats[f"{name}_size_mb"] = total_size / (1024 * 1024)

        return stats

    def close(self):
        """Close database connection."""
        if self._db_conn:
            self._db_conn.close()
