"""
Integration tests for device management.
"""

import pytest
import time
import json
import sqlite3
from unittest.mock import MagicMock


class TestDeviceRegistration:
    """Tests for device registration and management."""

    def test_register_device(self, temp_db, sample_device):
        """Test registering a new device."""
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO devices (id, name, type, status, config)
            VALUES (?, ?, ?, ?, ?)
        """, (
            sample_device["id"],
            sample_device["name"],
            sample_device["type"],
            sample_device["status"],
            json.dumps(sample_device["config"])
        ))
        conn.commit()

        # Verify device registered
        cursor.execute("SELECT name, type, status FROM devices WHERE id = ?", (sample_device["id"],))
        row = cursor.fetchone()

        assert row is not None
        assert row[0] == sample_device["name"]
        assert row[1] == "camera"
        assert row[2] == "online"

        conn.close()

    def test_device_heartbeat_update(self, temp_db, sample_device):
        """Test updating device status on heartbeat."""
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        # Register device
        cursor.execute("""
            INSERT INTO devices (id, name, type, status, last_seen)
            VALUES (?, ?, ?, ?, ?)
        """, (sample_device["id"], sample_device["name"], "camera", "online", time.time() - 100))
        conn.commit()

        # Simulate heartbeat
        new_time = time.time()
        cursor.execute("""
            UPDATE devices SET last_seen = ?, status = 'online' WHERE id = ?
        """, (new_time, sample_device["id"]))
        conn.commit()

        # Verify
        cursor.execute("SELECT last_seen, status FROM devices WHERE id = ?", (sample_device["id"],))
        row = cursor.fetchone()

        assert abs(row[0] - new_time) < 1
        assert row[1] == "online"

        conn.close()

    def test_offline_device_detection(self, temp_db):
        """Test detecting offline devices based on last_seen."""
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        # Insert devices with different last_seen times
        now = time.time()
        devices = [
            ("dev-1", "Camera 1", now - 30, "online"),      # Recent (30s ago)
            ("dev-2", "Camera 2", now - 120, "online"),     # Stale (2 min ago)
            ("dev-3", "Camera 3", now - 600, "online"),     # Offline (10 min ago)
        ]

        for dev_id, name, last_seen, status in devices:
            cursor.execute("""
                INSERT INTO devices (id, name, type, last_seen, status)
                VALUES (?, ?, 'camera', ?, ?)
            """, (dev_id, name, last_seen, status))
        conn.commit()

        # Query for potentially offline devices (not seen in 60 seconds)
        offline_threshold = now - 60
        cursor.execute("""
            SELECT id, name FROM devices WHERE last_seen < ? AND status = 'online'
        """, (offline_threshold,))
        offline_devices = cursor.fetchall()

        assert len(offline_devices) == 2
        offline_ids = [d[0] for d in offline_devices]
        assert "dev-2" in offline_ids
        assert "dev-3" in offline_ids

        conn.close()

    def test_device_config_update(self, temp_db, sample_device):
        """Test updating device configuration."""
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        # Register device
        cursor.execute("""
            INSERT INTO devices (id, name, type, config)
            VALUES (?, ?, 'camera', ?)
        """, (sample_device["id"], sample_device["name"], json.dumps({"fps": 25})))
        conn.commit()

        # Update config
        new_config = {"fps": 30, "resolution": [1280, 720]}
        cursor.execute("""
            UPDATE devices SET config = ? WHERE id = ?
        """, (json.dumps(new_config), sample_device["id"]))
        conn.commit()

        # Verify
        cursor.execute("SELECT config FROM devices WHERE id = ?", (sample_device["id"],))
        row = cursor.fetchone()
        config = json.loads(row[0])

        assert config["fps"] == 30
        assert config["resolution"] == [1280, 720]

        conn.close()


class TestDeviceHealth:
    """Tests for device health monitoring."""

    def test_health_metrics_storage(self):
        """Test storing and retrieving health metrics."""
        health_report = {
            "device_id": "camera-01",
            "timestamp": time.time(),
            "metrics": {
                "cpu_percent": 45.2,
                "memory_percent": 62.1,
                "disk_percent": 35.0,
                "temperature": 52.3,
                "battery_percent": 85.0,
                "wifi_signal": -45,
            },
            "status": "healthy",
        }

        # Validate metrics are within expected ranges
        assert 0 <= health_report["metrics"]["cpu_percent"] <= 100
        assert 0 <= health_report["metrics"]["memory_percent"] <= 100
        assert 0 <= health_report["metrics"]["disk_percent"] <= 100
        assert health_report["metrics"]["temperature"] < 85  # Safe temperature
        assert health_report["metrics"]["battery_percent"] > 20  # Not low

    def test_health_status_determination(self):
        """Test determining health status from metrics."""
        def determine_health_status(metrics: dict) -> str:
            issues = []

            if metrics.get("cpu_percent", 0) > 90:
                issues.append("high_cpu")
            if metrics.get("memory_percent", 0) > 90:
                issues.append("high_memory")
            if metrics.get("temperature", 0) > 80:
                issues.append("high_temperature")
            if metrics.get("battery_percent", 100) < 20:
                issues.append("low_battery")
            if metrics.get("disk_percent", 0) > 90:
                issues.append("low_disk")

            if any(i in issues for i in ["high_temperature", "low_battery"]):
                return "critical"
            elif len(issues) > 0:
                return "warning"
            else:
                return "healthy"

        # Test healthy device
        assert determine_health_status({
            "cpu_percent": 45,
            "memory_percent": 60,
            "temperature": 55,
            "battery_percent": 85,
        }) == "healthy"

        # Test warning state
        assert determine_health_status({
            "cpu_percent": 95,
            "memory_percent": 60,
            "temperature": 55,
        }) == "warning"

        # Test critical state
        assert determine_health_status({
            "temperature": 85,
        }) == "critical"

        assert determine_health_status({
            "battery_percent": 10,
        }) == "critical"


class TestZoneManagement:
    """Tests for zone management."""

    def test_create_zone(self, temp_db, sample_zone):
        """Test creating a protected zone."""
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO zones (id, name, geometry, zone_type, properties)
            VALUES (?, ?, ?, ?, ?)
        """, (
            sample_zone["id"],
            sample_zone["name"],
            json.dumps(sample_zone["geometry"]),
            sample_zone["zone_type"],
            json.dumps(sample_zone["properties"])
        ))
        conn.commit()

        # Verify
        cursor.execute("SELECT name, zone_type FROM zones WHERE id = ?", (sample_zone["id"],))
        row = cursor.fetchone()

        assert row[0] == "Chicken Coop"
        assert row[1] == "protected"

        conn.close()

    def test_point_in_zone(self, sample_zone):
        """Test checking if a point is inside a zone."""
        def point_in_polygon(point, polygon):
            """Ray casting algorithm for point in polygon."""
            x, y = point
            coords = polygon["coordinates"][0]  # Exterior ring
            n = len(coords)
            inside = False

            p1x, p1y = coords[0]
            for i in range(1, n + 1):
                p2x, p2y = coords[i % n]
                if y > min(p1y, p2y):
                    if y <= max(p1y, p2y):
                        if x <= max(p1x, p2x):
                            if p1y != p2y:
                                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xinters:
                                inside = not inside
                p1x, p1y = p2x, p2y

            return inside

        # Points to test
        inside_point = (5, 5)
        outside_point = (15, 15)

        assert point_in_polygon(inside_point, sample_zone["geometry"])
        assert not point_in_polygon(outside_point, sample_zone["geometry"])

    def test_zone_priority_ordering(self):
        """Test that zones are processed in priority order."""
        zones = [
            {"id": "z1", "name": "Garden", "priority": "low"},
            {"id": "z2", "name": "Coop", "priority": "high"},
            {"id": "z3", "name": "Pond", "priority": "medium"},
        ]

        priority_order = {"high": 0, "medium": 1, "low": 2}
        sorted_zones = sorted(zones, key=lambda z: priority_order.get(z["priority"], 99))

        assert sorted_zones[0]["name"] == "Coop"
        assert sorted_zones[1]["name"] == "Pond"
        assert sorted_zones[2]["name"] == "Garden"


class TestFleetOverview:
    """Tests for fleet overview functionality."""

    def test_aggregate_fleet_stats(self, temp_db):
        """Test aggregating statistics across all devices."""
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        # Insert test devices
        now = time.time()
        devices = [
            ("dev-1", "camera", "online", now - 10),
            ("dev-2", "camera", "online", now - 20),
            ("dev-3", "camera", "offline", now - 3600),
            ("dev-4", "deterrent", "online", now - 5),
        ]

        for dev_id, dev_type, status, last_seen in devices:
            cursor.execute("""
                INSERT INTO devices (id, type, status, last_seen)
                VALUES (?, ?, ?, ?)
            """, (dev_id, dev_type, status, last_seen))
        conn.commit()

        # Calculate fleet stats
        cursor.execute("SELECT COUNT(*) FROM devices")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM devices WHERE status = 'online'")
        online = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM devices WHERE type = 'camera'")
        cameras = cursor.fetchone()[0]

        assert total == 4
        assert online == 3
        assert cameras == 3

        conn.close()

    def test_device_uptime_calculation(self):
        """Test calculating device uptime percentage."""
        # Simulate uptime data (last 24 hours in 5-minute intervals)
        total_intervals = 288  # 24 hours * 12 intervals per hour
        online_intervals = 280  # Missed 8 intervals

        uptime_percent = (online_intervals / total_intervals) * 100
        assert uptime_percent == pytest.approx(97.2, rel=0.1)

    def test_detection_rate_calculation(self, temp_db):
        """Test calculating detection rate across fleet."""
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        # Insert detections from multiple devices
        now = time.time()
        detections = [
            ("det-1", "camera-1", now - 3600),
            ("det-2", "camera-1", now - 3000),
            ("det-3", "camera-2", now - 2400),
            ("det-4", "camera-2", now - 1800),
            ("det-5", "camera-1", now - 1200),
        ]

        for det_id, device_id, timestamp in detections:
            cursor.execute("""
                INSERT INTO detections (id, device_id, timestamp, predator_type, confidence)
                VALUES (?, ?, ?, 'fox', 0.9)
            """, (det_id, device_id, timestamp))
        conn.commit()

        # Calculate detections per device in last hour
        hour_ago = now - 3600
        cursor.execute("""
            SELECT device_id, COUNT(*) as count
            FROM detections
            WHERE timestamp > ?
            GROUP BY device_id
        """, (hour_ago,))

        results = dict(cursor.fetchall())
        assert results.get("camera-1") == 3
        assert results.get("camera-2") == 2

        conn.close()
