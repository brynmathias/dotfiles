"""
Event store for historical data.

Stores and retrieves events for replay functionality.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Iterator
from enum import Enum
from pathlib import Path
import json
import sqlite3
import time
import logging

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of events that can be stored."""
    DETECTION = "detection"
    ALERT = "alert"
    DETERRENCE = "deterrence"
    DEVICE_STATUS = "device_status"
    TRACK_UPDATE = "track_update"
    WEATHER_UPDATE = "weather"
    FRAME = "frame"  # Reference to frame, not actual data
    SYSTEM = "system"


@dataclass
class StoredEvent:
    """An event stored in the event store."""
    id: str
    event_type: EventType
    timestamp: float
    device_id: Optional[str]
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "device_id": self.device_id,
            "data": self.data,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoredEvent":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            event_type=EventType(data["event_type"]),
            timestamp=data["timestamp"],
            device_id=data.get("device_id"),
            data=data.get("data", {}),
            metadata=data.get("metadata", {}),
        )


class EventStore:
    """
    Persistent event store for replay functionality.

    Stores events in SQLite with efficient time-range queries.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                device_id TEXT,
                data TEXT NOT NULL,
                metadata TEXT DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_events_timestamp
                ON events(timestamp);

            CREATE INDEX IF NOT EXISTS idx_events_type_timestamp
                ON events(event_type, timestamp);

            CREATE INDEX IF NOT EXISTS idx_events_device_timestamp
                ON events(device_id, timestamp);

            -- Frame references table
            CREATE TABLE IF NOT EXISTS frame_refs (
                id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                device_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                frame_path TEXT NOT NULL,
                thumbnail_path TEXT,
                FOREIGN KEY (event_id) REFERENCES events(id)
            );

            CREATE INDEX IF NOT EXISTS idx_frames_timestamp
                ON frame_refs(timestamp);

            -- Retention metadata
            CREATE TABLE IF NOT EXISTS retention_info (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        conn.commit()
        conn.close()

    def store_event(self, event: StoredEvent) -> bool:
        """
        Store an event.

        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT OR REPLACE INTO events
                (id, event_type, timestamp, device_id, data, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                event.id,
                event.event_type.value,
                event.timestamp,
                event.device_id,
                json.dumps(event.data),
                json.dumps(event.metadata),
            ))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Failed to store event: {e}")
            return False

    def store_events_batch(self, events: List[StoredEvent]) -> int:
        """
        Store multiple events in a batch.

        Returns:
            Number of events stored
        """
        if not events:
            return 0

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.executemany("""
                INSERT OR REPLACE INTO events
                (id, event_type, timestamp, device_id, data, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [
                (
                    e.id,
                    e.event_type.value,
                    e.timestamp,
                    e.device_id,
                    json.dumps(e.data),
                    json.dumps(e.metadata),
                )
                for e in events
            ])

            conn.commit()
            count = cursor.rowcount
            conn.close()
            return count
        except Exception as e:
            logger.error(f"Failed to store events batch: {e}")
            return 0

    def get_event(self, event_id: str) -> Optional[StoredEvent]:
        """Get a single event by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT id, event_type, timestamp, device_id, data, metadata FROM events WHERE id = ?",
            (event_id,)
        )
        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None

        return StoredEvent(
            id=row[0],
            event_type=EventType(row[1]),
            timestamp=row[2],
            device_id=row[3],
            data=json.loads(row[4]),
            metadata=json.loads(row[5]) if row[5] else {},
        )

    def get_events_in_range(
        self,
        start_time: float,
        end_time: float,
        event_types: Optional[List[EventType]] = None,
        device_ids: Optional[List[str]] = None,
        limit: int = 10000,
    ) -> List[StoredEvent]:
        """
        Get events within a time range.

        Args:
            start_time: Start timestamp (inclusive)
            end_time: End timestamp (inclusive)
            event_types: Optional filter by event types
            device_ids: Optional filter by device IDs
            limit: Maximum number of events to return

        Returns:
            List of events ordered by timestamp
        """
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT id, event_type, timestamp, device_id, data, metadata
            FROM events
            WHERE timestamp >= ? AND timestamp <= ?
        """
        params: List[Any] = [start_time, end_time]

        if event_types:
            placeholders = ",".join("?" * len(event_types))
            query += f" AND event_type IN ({placeholders})"
            params.extend(et.value for et in event_types)

        if device_ids:
            placeholders = ",".join("?" * len(device_ids))
            query += f" AND device_id IN ({placeholders})"
            params.extend(device_ids)

        query += " ORDER BY timestamp ASC LIMIT ?"
        params.append(limit)

        cursor = conn.execute(query, params)
        events = [
            StoredEvent(
                id=row[0],
                event_type=EventType(row[1]),
                timestamp=row[2],
                device_id=row[3],
                data=json.loads(row[4]),
                metadata=json.loads(row[5]) if row[5] else {},
            )
            for row in cursor.fetchall()
        ]
        conn.close()

        return events

    def iterate_events(
        self,
        start_time: float,
        end_time: float,
        event_types: Optional[List[EventType]] = None,
        batch_size: int = 1000,
    ) -> Iterator[StoredEvent]:
        """
        Iterate over events in a time range.

        Uses batched queries for memory efficiency with large datasets.
        """
        current_time = start_time

        while current_time <= end_time:
            batch = self.get_events_in_range(
                start_time=current_time,
                end_time=end_time,
                event_types=event_types,
                limit=batch_size,
            )

            if not batch:
                break

            for event in batch:
                yield event

            # Move to next batch
            current_time = batch[-1].timestamp + 0.0001  # Small epsilon

    def get_timeline_summary(
        self,
        start_time: float,
        end_time: float,
        bucket_seconds: int = 300,  # 5-minute buckets
    ) -> List[Dict[str, Any]]:
        """
        Get a summary of events grouped into time buckets.

        Useful for displaying a timeline overview.
        """
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT
                (CAST(timestamp / ? AS INTEGER) * ?) as bucket,
                event_type,
                COUNT(*) as count
            FROM events
            WHERE timestamp >= ? AND timestamp <= ?
            GROUP BY bucket, event_type
            ORDER BY bucket
        """

        cursor = conn.execute(query, (bucket_seconds, bucket_seconds, start_time, end_time))

        # Aggregate results
        buckets: Dict[float, Dict[str, int]] = {}
        for row in cursor.fetchall():
            bucket_time, event_type, count = row
            if bucket_time not in buckets:
                buckets[bucket_time] = {}
            buckets[bucket_time][event_type] = count

        conn.close()

        # Format results
        return [
            {
                "timestamp": bucket_time,
                "events": counts,
                "total": sum(counts.values()),
            }
            for bucket_time, counts in sorted(buckets.items())
        ]

    def get_detection_events(
        self,
        start_time: float,
        end_time: float,
        predator_types: Optional[List[str]] = None,
    ) -> List[StoredEvent]:
        """Get detection events with optional predator type filter."""
        events = self.get_events_in_range(
            start_time=start_time,
            end_time=end_time,
            event_types=[EventType.DETECTION],
        )

        if predator_types:
            events = [
                e for e in events
                if e.data.get("predator_type") in predator_types
            ]

        return events

    def store_frame_reference(
        self,
        event_id: str,
        device_id: str,
        timestamp: float,
        frame_path: str,
        thumbnail_path: Optional[str] = None,
    ) -> bool:
        """Store a reference to a video frame."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT OR REPLACE INTO frame_refs
                (id, event_id, device_id, timestamp, frame_path, thumbnail_path)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                f"frame-{timestamp}-{device_id}",
                event_id,
                device_id,
                timestamp,
                frame_path,
                thumbnail_path,
            ))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Failed to store frame reference: {e}")
            return False

    def get_frames_in_range(
        self,
        start_time: float,
        end_time: float,
        device_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get frame references in a time range."""
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT id, event_id, device_id, timestamp, frame_path, thumbnail_path
            FROM frame_refs
            WHERE timestamp >= ? AND timestamp <= ?
        """
        params: List[Any] = [start_time, end_time]

        if device_id:
            query += " AND device_id = ?"
            params.append(device_id)

        query += " ORDER BY timestamp ASC"

        cursor = conn.execute(query, params)
        frames = [
            {
                "id": row[0],
                "event_id": row[1],
                "device_id": row[2],
                "timestamp": row[3],
                "frame_path": row[4],
                "thumbnail_path": row[5],
            }
            for row in cursor.fetchall()
        ]
        conn.close()

        return frames

    def cleanup_old_events(self, retention_days: int = 30) -> int:
        """
        Remove events older than retention period.

        Returns:
            Number of events deleted
        """
        cutoff = time.time() - (retention_days * 86400)

        conn = sqlite3.connect(self.db_path)

        # Delete frame references first (foreign key)
        conn.execute("DELETE FROM frame_refs WHERE timestamp < ?", (cutoff,))

        cursor = conn.execute("DELETE FROM events WHERE timestamp < ?", (cutoff,))
        deleted = cursor.rowcount

        conn.commit()
        conn.close()

        logger.info(f"Cleaned up {deleted} events older than {retention_days} days")
        return deleted

    def get_stats(self) -> Dict[str, Any]:
        """Get event store statistics."""
        conn = sqlite3.connect(self.db_path)

        cursor = conn.execute("SELECT COUNT(*) FROM events")
        total_events = cursor.fetchone()[0]

        cursor = conn.execute("SELECT MIN(timestamp), MAX(timestamp) FROM events")
        row = cursor.fetchone()
        min_time, max_time = row[0], row[1]

        cursor = conn.execute("""
            SELECT event_type, COUNT(*) as count
            FROM events
            GROUP BY event_type
        """)
        by_type = dict(cursor.fetchall())

        conn.close()

        return {
            "total_events": total_events,
            "earliest_event": min_time,
            "latest_event": max_time,
            "events_by_type": by_type,
        }
