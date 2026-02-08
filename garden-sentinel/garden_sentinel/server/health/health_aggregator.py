"""
Server-side health aggregation for all edge devices.

Collects health reports from edge devices and provides:
- Fleet-wide health overview
- Historical health data
- Alert aggregation
- Predictive maintenance warnings
"""

import asyncio
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Callable, Awaitable

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Overall health status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


@dataclass
class DeviceHealth:
    """Health status for a single device."""
    device_id: str
    last_seen: datetime
    overall_status: HealthStatus = HealthStatus.UNKNOWN

    # Battery
    battery_voltage: Optional[float] = None
    battery_percentage: Optional[float] = None
    battery_is_charging: bool = False
    battery_time_remaining_min: Optional[float] = None

    # System
    cpu_temp: Optional[float] = None
    cpu_usage: Optional[float] = None
    memory_used_percent: Optional[float] = None
    disk_used_percent: Optional[float] = None
    is_throttled: bool = False

    # Network
    is_connected: bool = False
    server_latency_ms: Optional[float] = None
    wifi_signal_percent: Optional[float] = None

    # Camera
    camera_available: bool = False
    camera_fps: Optional[float] = None

    # Alerts
    alerts: list[str] = field(default_factory=list)

    @property
    def is_online(self) -> bool:
        """Check if device is online (seen within last 60 seconds)."""
        return (datetime.now() - self.last_seen).total_seconds() < 60

    def to_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "last_seen": self.last_seen.isoformat(),
            "is_online": self.is_online,
            "overall_status": self.overall_status.value,
            "battery": {
                "voltage": self.battery_voltage,
                "percentage": self.battery_percentage,
                "is_charging": self.battery_is_charging,
                "time_remaining_min": self.battery_time_remaining_min,
            } if self.battery_voltage else None,
            "system": {
                "cpu_temp": self.cpu_temp,
                "cpu_usage": self.cpu_usage,
                "memory_used_percent": self.memory_used_percent,
                "disk_used_percent": self.disk_used_percent,
                "is_throttled": self.is_throttled,
            },
            "network": {
                "is_connected": self.is_connected,
                "latency_ms": self.server_latency_ms,
                "wifi_signal_percent": self.wifi_signal_percent,
            },
            "camera": {
                "available": self.camera_available,
                "fps": self.camera_fps,
            },
            "alerts": self.alerts,
        }


@dataclass
class FleetHealth:
    """Aggregated health for all devices."""
    total_devices: int = 0
    online_devices: int = 0
    healthy_devices: int = 0
    warning_devices: int = 0
    critical_devices: int = 0
    offline_devices: int = 0

    # Aggregated metrics
    avg_battery_percentage: Optional[float] = None
    min_battery_percentage: Optional[float] = None
    lowest_battery_device: Optional[str] = None

    avg_cpu_temp: Optional[float] = None
    max_cpu_temp: Optional[float] = None
    hottest_device: Optional[str] = None

    total_alerts: int = 0
    devices: list[DeviceHealth] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_devices": self.total_devices,
            "online_devices": self.online_devices,
            "healthy_devices": self.healthy_devices,
            "warning_devices": self.warning_devices,
            "critical_devices": self.critical_devices,
            "offline_devices": self.offline_devices,
            "battery": {
                "avg_percentage": self.avg_battery_percentage,
                "min_percentage": self.min_battery_percentage,
                "lowest_device": self.lowest_battery_device,
            },
            "temperature": {
                "avg_cpu_temp": self.avg_cpu_temp,
                "max_cpu_temp": self.max_cpu_temp,
                "hottest_device": self.hottest_device,
            },
            "total_alerts": self.total_alerts,
            "devices": [d.to_dict() for d in self.devices],
        }


class HealthAggregator:
    """
    Aggregates health reports from all edge devices.

    Features:
    - Stores health history in SQLite
    - Provides fleet-wide health overview
    - Detects devices going offline
    - Generates alerts for health issues
    """

    def __init__(
        self,
        db_path: Path,
        offline_threshold_seconds: float = 60.0,
        history_retention_days: int = 7,
    ):
        self.db_path = db_path
        self.offline_threshold_seconds = offline_threshold_seconds
        self.history_retention_days = history_retention_days

        # Current device states
        self._devices: dict[str, DeviceHealth] = {}

        # Callbacks
        self._on_device_offline: Optional[Callable[[str], Awaitable[None]]] = None
        self._on_health_alert: Optional[Callable[[str, str, HealthStatus], Awaitable[None]]] = None

        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._offline_check_task: Optional[asyncio.Task] = None

        self._init_db()

    def _init_db(self):
        """Initialize the health history database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS health_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    device_id TEXT NOT NULL,
                    overall_status TEXT,
                    battery_voltage REAL,
                    battery_percentage REAL,
                    cpu_temp REAL,
                    cpu_usage REAL,
                    memory_used REAL,
                    disk_used REAL,
                    is_throttled BOOLEAN,
                    wifi_signal REAL,
                    camera_fps REAL,
                    alerts TEXT
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_health_timestamp
                ON health_history(timestamp)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_health_device
                ON health_history(device_id)
            """)

    def set_callbacks(
        self,
        on_device_offline: Optional[Callable[[str], Awaitable[None]]] = None,
        on_health_alert: Optional[Callable[[str, str, HealthStatus], Awaitable[None]]] = None,
    ):
        """Set event callbacks."""
        self._on_device_offline = on_device_offline
        self._on_health_alert = on_health_alert

    async def start(self):
        """Start the health aggregator."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._offline_check_task = asyncio.create_task(self._offline_check_loop())
        logger.info("Health aggregator started")

    async def stop(self):
        """Stop the health aggregator."""
        for task in [self._cleanup_task, self._offline_check_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        logger.info("Health aggregator stopped")

    async def process_health_report(self, report: dict):
        """
        Process a health report from an edge device.

        Expected format matches HealthReport.to_dict() from edge device.
        """
        device_id = report.get("device_id")
        if not device_id:
            logger.warning("Received health report without device_id")
            return

        # Parse report into DeviceHealth
        device = DeviceHealth(
            device_id=device_id,
            last_seen=datetime.now(),
        )

        # Parse overall status
        status_str = report.get("overall_status", "unknown")
        try:
            device.overall_status = HealthStatus(status_str)
        except ValueError:
            device.overall_status = HealthStatus.UNKNOWN

        # Parse battery
        battery = report.get("battery")
        if battery:
            device.battery_voltage = battery.get("voltage")
            device.battery_percentage = battery.get("percentage")
            device.battery_is_charging = battery.get("is_charging", False)
            device.battery_time_remaining_min = battery.get("time_remaining_minutes")

        # Parse system
        system = report.get("system")
        if system:
            device.cpu_temp = system.get("cpu_temp")
            device.cpu_usage = system.get("cpu_usage")
            device.memory_used_percent = system.get("memory_used_percent")
            device.disk_used_percent = system.get("disk_used_percent")
            device.is_throttled = system.get("is_throttled", False)

        # Parse network
        network = report.get("network")
        if network:
            device.is_connected = network.get("is_connected", False)
            device.server_latency_ms = network.get("latency_ms")
            device.wifi_signal_percent = network.get("wifi_signal_percent")

        # Parse camera
        camera = report.get("camera")
        if camera:
            device.camera_available = camera.get("is_available", False)
            device.camera_fps = camera.get("fps")

        # Parse alerts
        device.alerts = report.get("alerts", [])

        # Check for new alerts
        old_device = self._devices.get(device_id)
        if old_device and self._on_health_alert:
            new_alerts = set(device.alerts) - set(old_device.alerts)
            for alert in new_alerts:
                await self._on_health_alert(device_id, alert, device.overall_status)

        # Update device state
        self._devices[device_id] = device

        # Store in history
        self._store_health_record(device)

        logger.debug(f"Processed health report from {device_id}: {device.overall_status.value}")

    def _store_health_record(self, device: DeviceHealth):
        """Store health record in database."""
        import json

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO health_history (
                    timestamp, device_id, overall_status,
                    battery_voltage, battery_percentage,
                    cpu_temp, cpu_usage, memory_used, disk_used,
                    is_throttled, wifi_signal, camera_fps, alerts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                device.last_seen.isoformat(),
                device.device_id,
                device.overall_status.value,
                device.battery_voltage,
                device.battery_percentage,
                device.cpu_temp,
                device.cpu_usage,
                device.memory_used_percent,
                device.disk_used_percent,
                device.is_throttled,
                device.wifi_signal_percent,
                device.camera_fps,
                json.dumps(device.alerts),
            ))

    def get_device_health(self, device_id: str) -> Optional[DeviceHealth]:
        """Get health status for a specific device."""
        return self._devices.get(device_id)

    def get_fleet_health(self) -> FleetHealth:
        """Get aggregated health for all devices."""
        fleet = FleetHealth()
        fleet.total_devices = len(self._devices)

        battery_percentages = []
        cpu_temps = []

        for device in self._devices.values():
            fleet.devices.append(device)

            if device.is_online:
                fleet.online_devices += 1
            else:
                fleet.offline_devices += 1
                continue  # Don't count offline devices in health stats

            if device.overall_status == HealthStatus.HEALTHY:
                fleet.healthy_devices += 1
            elif device.overall_status == HealthStatus.WARNING:
                fleet.warning_devices += 1
            elif device.overall_status == HealthStatus.CRITICAL:
                fleet.critical_devices += 1

            if device.battery_percentage is not None:
                battery_percentages.append((device.device_id, device.battery_percentage))

            if device.cpu_temp is not None:
                cpu_temps.append((device.device_id, device.cpu_temp))

            fleet.total_alerts += len(device.alerts)

        # Calculate battery stats
        if battery_percentages:
            percentages = [p[1] for p in battery_percentages]
            fleet.avg_battery_percentage = sum(percentages) / len(percentages)
            min_battery = min(battery_percentages, key=lambda x: x[1])
            fleet.min_battery_percentage = min_battery[1]
            fleet.lowest_battery_device = min_battery[0]

        # Calculate temperature stats
        if cpu_temps:
            temps = [t[1] for t in cpu_temps]
            fleet.avg_cpu_temp = sum(temps) / len(temps)
            max_temp = max(cpu_temps, key=lambda x: x[1])
            fleet.max_cpu_temp = max_temp[1]
            fleet.hottest_device = max_temp[0]

        return fleet

    def get_all_devices(self) -> list[DeviceHealth]:
        """Get all device health states."""
        return list(self._devices.values())

    async def get_device_history(
        self,
        device_id: str,
        hours: int = 24,
    ) -> list[dict]:
        """Get health history for a device."""
        import json

        cutoff = datetime.now() - timedelta(hours=hours)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM health_history
                WHERE device_id = ? AND timestamp > ?
                ORDER BY timestamp ASC
            """, (device_id, cutoff.isoformat()))

            history = []
            for row in cursor:
                history.append({
                    "timestamp": row["timestamp"],
                    "overall_status": row["overall_status"],
                    "battery_voltage": row["battery_voltage"],
                    "battery_percentage": row["battery_percentage"],
                    "cpu_temp": row["cpu_temp"],
                    "cpu_usage": row["cpu_usage"],
                    "memory_used": row["memory_used"],
                    "disk_used": row["disk_used"],
                    "is_throttled": bool(row["is_throttled"]),
                    "wifi_signal": row["wifi_signal"],
                    "camera_fps": row["camera_fps"],
                    "alerts": json.loads(row["alerts"]) if row["alerts"] else [],
                })

            return history

    async def _offline_check_loop(self):
        """Periodically check for devices going offline."""
        while True:
            try:
                await asyncio.sleep(15)  # Check every 15 seconds

                now = datetime.now()
                for device_id, device in list(self._devices.items()):
                    was_online = device.is_online
                    time_since_seen = (now - device.last_seen).total_seconds()

                    if time_since_seen > self.offline_threshold_seconds:
                        if was_online or device.overall_status != HealthStatus.OFFLINE:
                            device.overall_status = HealthStatus.OFFLINE
                            logger.warning(f"Device {device_id} went offline")

                            if self._on_device_offline:
                                await self._on_device_offline(device_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in offline check: {e}")

    async def _cleanup_loop(self):
        """Periodically cleanup old health records."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run hourly

                cutoff = datetime.now() - timedelta(days=self.history_retention_days)

                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("""
                        DELETE FROM health_history
                        WHERE timestamp < ?
                    """, (cutoff.isoformat(),))

                    if cursor.rowcount > 0:
                        logger.info(f"Cleaned up {cursor.rowcount} old health records")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup: {e}")
