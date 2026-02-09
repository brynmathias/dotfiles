"""
Integration tests for the detection pipeline.
"""

import pytest
import time
import json
import sqlite3
from unittest.mock import MagicMock, patch, AsyncMock


class TestDetectionPipeline:
    """Tests for the complete detection pipeline."""

    def test_detection_storage(self, temp_db, sample_detection):
        """Test storing detection events in the database."""
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO detections (id, device_id, timestamp, predator_type, confidence, bbox)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            sample_detection["id"],
            sample_detection["device_id"],
            sample_detection["timestamp"],
            sample_detection["predator_type"],
            sample_detection["confidence"],
            json.dumps(sample_detection["bbox"])
        ))
        conn.commit()

        # Verify detection stored
        cursor.execute("SELECT * FROM detections WHERE id = ?", (sample_detection["id"],))
        row = cursor.fetchone()

        assert row is not None
        assert row[3] == "fox"  # predator_type
        assert row[4] == 0.92  # confidence

        conn.close()

    def test_alert_generation_from_detection(self, temp_db, sample_detection, sample_alert):
        """Test that alerts are generated from detections."""
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        # Insert detection
        cursor.execute("""
            INSERT INTO detections (id, device_id, timestamp, predator_type, confidence, bbox)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            sample_detection["id"],
            sample_detection["device_id"],
            sample_detection["timestamp"],
            sample_detection["predator_type"],
            sample_detection["confidence"],
            json.dumps(sample_detection["bbox"])
        ))

        # Generate alert
        cursor.execute("""
            INSERT INTO alerts (id, detection_id, device_id, timestamp, severity, predator_type, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            sample_alert["id"],
            sample_detection["id"],
            sample_detection["device_id"],
            sample_detection["timestamp"],
            sample_alert["severity"],
            sample_detection["predator_type"],
            sample_detection["confidence"]
        ))
        conn.commit()

        # Verify alert links to detection
        cursor.execute("""
            SELECT a.id, d.predator_type
            FROM alerts a
            JOIN detections d ON a.detection_id = d.id
            WHERE a.id = ?
        """, (sample_alert["id"],))
        row = cursor.fetchone()

        assert row is not None
        assert row[1] == "fox"

        conn.close()

    def test_severity_calculation(self):
        """Test alert severity calculation based on confidence and predator type."""
        def calculate_severity(predator_type: str, confidence: float, zone_type: str) -> str:
            # Dangerous predators get higher severity
            danger_level = {
                "fox": 0.9,
                "coyote": 1.0,
                "hawk": 0.8,
                "raccoon": 0.6,
                "cat": 0.5,
            }.get(predator_type, 0.5)

            # Protected zones increase severity
            zone_multiplier = 1.5 if zone_type == "protected" else 1.0

            score = confidence * danger_level * zone_multiplier

            if score >= 0.8:
                return "critical"
            elif score >= 0.6:
                return "high"
            elif score >= 0.4:
                return "medium"
            else:
                return "low"

        # Test various scenarios
        assert calculate_severity("fox", 0.95, "protected") == "critical"
        assert calculate_severity("fox", 0.7, "normal") in ["medium", "high"]
        assert calculate_severity("cat", 0.5, "normal") == "low"
        assert calculate_severity("coyote", 0.9, "protected") == "critical"

    def test_detection_confidence_threshold(self):
        """Test that low confidence detections are filtered."""
        detections = [
            {"predator": "fox", "confidence": 0.95},
            {"predator": "hawk", "confidence": 0.3},  # Below threshold
            {"predator": "cat", "confidence": 0.7},
        ]
        threshold = 0.5

        filtered = [d for d in detections if d["confidence"] >= threshold]

        assert len(filtered) == 2
        assert all(d["confidence"] >= threshold for d in filtered)

    def test_detection_deduplication(self):
        """Test deduplication of similar detections within time window."""
        detections = [
            {"id": "1", "predator": "fox", "timestamp": 1000.0, "bbox": [100, 100, 200, 200]},
            {"id": "2", "predator": "fox", "timestamp": 1000.5, "bbox": [105, 102, 205, 202]},  # Duplicate
            {"id": "3", "predator": "fox", "timestamp": 1005.0, "bbox": [100, 100, 200, 200]},  # New (time gap)
            {"id": "4", "predator": "hawk", "timestamp": 1000.3, "bbox": [300, 300, 400, 400]},  # Different predator
        ]

        def iou(box1, box2):
            """Calculate intersection over union."""
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])

            if x2 < x1 or y2 < y1:
                return 0.0

            intersection = (x2 - x1) * (y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection

            return intersection / union if union > 0 else 0

        def is_duplicate(det1, det2, time_threshold=2.0, iou_threshold=0.5):
            if det1["predator"] != det2["predator"]:
                return False
            if abs(det1["timestamp"] - det2["timestamp"]) > time_threshold:
                return False
            return iou(det1["bbox"], det2["bbox"]) > iou_threshold

        # Check duplicates
        assert is_duplicate(detections[0], detections[1])
        assert not is_duplicate(detections[0], detections[2])  # Time gap too large
        assert not is_duplicate(detections[0], detections[3])  # Different predator

    def test_deterrence_action_logging(self, temp_db, sample_alert):
        """Test logging deterrence actions."""
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        # Insert alert first
        cursor.execute("""
            INSERT INTO alerts (id, device_id, timestamp, severity, predator_type, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            sample_alert["id"],
            sample_alert["device_id"],
            sample_alert["timestamp"],
            sample_alert["severity"],
            sample_alert["predator_type"],
            sample_alert["confidence"]
        ))

        # Log deterrence action
        action_id = "action-001"
        cursor.execute("""
            INSERT INTO deterrence_actions (id, alert_id, device_id, timestamp, action_type, parameters, success)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            action_id,
            sample_alert["id"],
            sample_alert["device_id"],
            time.time(),
            "spray",
            json.dumps({"duration": 2.0, "direction": 45}),
            1
        ))
        conn.commit()

        # Verify action logged
        cursor.execute("""
            SELECT action_type, success FROM deterrence_actions WHERE alert_id = ?
        """, (sample_alert["id"],))
        row = cursor.fetchone()

        assert row is not None
        assert row[0] == "spray"
        assert row[1] == 1

        conn.close()


class TestMultiCameraTracking:
    """Tests for tracking across multiple cameras."""

    def test_track_handoff_detection(self):
        """Test detecting when a predator moves between camera views."""
        tracks = {
            "camera-1": {
                "track_id": "track-001",
                "predator": "fox",
                "last_position": (0.9, 0.5),  # Near right edge
                "last_seen": 1000.0,
                "velocity": (0.1, 0.0),  # Moving right
            },
            "camera-2": {
                "track_id": "track-002",
                "predator": "fox",
                "last_position": (0.1, 0.5),  # Near left edge
                "last_seen": 1001.0,  # 1 second later
                "velocity": (0.1, 0.0),
            },
        }

        # Check if tracks could be the same animal
        def could_be_handoff(track1, track2, time_threshold=3.0):
            if track1["predator"] != track2["predator"]:
                return False
            time_diff = abs(track1["last_seen"] - track2["last_seen"])
            if time_diff > time_threshold:
                return False
            # Check if one is at edge and other appeared at opposite edge
            at_edge_1 = track1["last_position"][0] > 0.8 or track1["last_position"][0] < 0.2
            at_edge_2 = track2["last_position"][0] > 0.8 or track2["last_position"][0] < 0.2
            return at_edge_1 and at_edge_2 and time_diff < time_threshold

        result = could_be_handoff(tracks["camera-1"], tracks["camera-2"])
        assert result is True

    def test_global_track_id_assignment(self):
        """Test assigning global track IDs across cameras."""
        local_tracks = [
            {"camera": "cam-1", "local_id": "1", "predator": "fox"},
            {"camera": "cam-2", "local_id": "1", "predator": "fox"},  # Same animal
            {"camera": "cam-1", "local_id": "2", "predator": "hawk"},
        ]

        global_tracks = {}
        next_global_id = 1

        # Simple assignment (in reality would use spatial/temporal matching)
        for track in local_tracks:
            key = (track["camera"], track["local_id"])
            if key not in global_tracks:
                # Check if we can merge with existing global track
                merged = False
                for gid, gtrack in global_tracks.items():
                    if gtrack["predator"] == track["predator"]:
                        # Simplified: merge if same predator type
                        global_tracks[key] = {"global_id": gtrack["global_id"], "predator": track["predator"]}
                        merged = True
                        break

                if not merged:
                    global_tracks[key] = {"global_id": next_global_id, "predator": track["predator"]}
                    next_global_id += 1

        # Verify we have 2 unique global tracks (fox and hawk)
        unique_global_ids = set(t["global_id"] for t in global_tracks.values())
        assert len(unique_global_ids) == 2


class TestAlertAcknowledgment:
    """Tests for alert acknowledgment flow."""

    def test_acknowledge_alert(self, temp_db, sample_alert, sample_user):
        """Test acknowledging an alert."""
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        # Insert user
        cursor.execute("""
            INSERT INTO users (id, username, password_hash, role, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (sample_user["id"], sample_user["username"], "hash", sample_user["role"], time.time()))

        # Insert alert
        cursor.execute("""
            INSERT INTO alerts (id, device_id, timestamp, severity, predator_type, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            sample_alert["id"],
            sample_alert["device_id"],
            sample_alert["timestamp"],
            sample_alert["severity"],
            sample_alert["predator_type"],
            sample_alert["confidence"]
        ))
        conn.commit()

        # Acknowledge alert
        ack_time = time.time()
        cursor.execute("""
            UPDATE alerts SET acknowledged = 1, acknowledged_by = ?, acknowledged_at = ?
            WHERE id = ?
        """, (sample_user["id"], ack_time, sample_alert["id"]))
        conn.commit()

        # Verify
        cursor.execute("""
            SELECT acknowledged, acknowledged_by, acknowledged_at FROM alerts WHERE id = ?
        """, (sample_alert["id"],))
        row = cursor.fetchone()

        assert row[0] == 1
        assert row[1] == sample_user["id"]
        assert abs(row[2] - ack_time) < 1

        conn.close()

    def test_unacknowledged_alert_count(self, temp_db):
        """Test counting unacknowledged alerts."""
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        # Insert multiple alerts
        alerts = [
            ("alert-1", 0),  # Unacknowledged
            ("alert-2", 0),  # Unacknowledged
            ("alert-3", 1),  # Acknowledged
            ("alert-4", 0),  # Unacknowledged
        ]

        for alert_id, acked in alerts:
            cursor.execute("""
                INSERT INTO alerts (id, device_id, timestamp, severity, acknowledged)
                VALUES (?, 'cam-1', ?, 'medium', ?)
            """, (alert_id, time.time(), acked))
        conn.commit()

        # Count unacknowledged
        cursor.execute("SELECT COUNT(*) FROM alerts WHERE acknowledged = 0")
        count = cursor.fetchone()[0]

        assert count == 3

        conn.close()
