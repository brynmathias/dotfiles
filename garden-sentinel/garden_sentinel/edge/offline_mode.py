"""
Offline mode handler for edge devices.
Enables autonomous operation when server connection is lost.
"""

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class OfflineConfig:
    enabled: bool = True
    # How long to wait before switching to offline mode
    connection_timeout_s: float = 10.0
    # How often to check connection
    heartbeat_interval_s: float = 5.0
    # Queue alerts to send when back online
    queue_alerts: bool = True
    max_queued_alerts: int = 100
    # Local storage for offline events
    offline_storage_dir: str = "/var/lib/garden-sentinel/offline"
    # Auto-respond to threats in offline mode
    auto_respond: bool = True


@dataclass
class QueuedAlert:
    timestamp: float
    device_id: str
    detections: list
    threat_level: str
    actions_taken: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "device_id": self.device_id,
            "detections": self.detections,
            "threat_level": self.threat_level,
            "actions_taken": self.actions_taken,
        }


class OfflineModeHandler:
    """
    Handles offline operation of edge devices.

    When server connection is lost:
    1. Switches to local inference (if available)
    2. Makes autonomous decisions based on threat level
    3. Queues alerts to send when reconnected
    4. Stores events locally
    """

    def __init__(
        self,
        config: OfflineConfig,
        device_id: str,
        edge_inference=None,
        gpio_controller=None,
        targeting_controller=None,
    ):
        self.config = config
        self.device_id = device_id
        self.edge_inference = edge_inference
        self.gpio = gpio_controller
        self.targeting = targeting_controller

        self._is_offline = False
        self._last_server_contact: float = time.time()
        self._alert_queue: queue.Queue[QueuedAlert] = queue.Queue(
            maxsize=config.max_queued_alerts
        )

        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None

        self._on_mode_change: Optional[Callable[[bool], None]] = None

        # Ensure offline storage directory exists
        Path(config.offline_storage_dir).mkdir(parents=True, exist_ok=True)

    def start(self):
        """Start offline mode monitoring."""
        if not self.config.enabled:
            return

        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_connection, daemon=True
        )
        self._monitor_thread.start()
        logger.info("Offline mode handler started")

    def stop(self):
        """Stop offline mode monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

    def server_contacted(self):
        """Call this when server communication succeeds."""
        self._last_server_contact = time.time()

        if self._is_offline:
            self._go_online()

    def _monitor_connection(self):
        """Monitor connection and switch modes as needed."""
        while self._running:
            try:
                time_since_contact = time.time() - self._last_server_contact

                if not self._is_offline and time_since_contact > self.config.connection_timeout_s:
                    self._go_offline()

                time.sleep(self.config.heartbeat_interval_s)

            except Exception as e:
                logger.error(f"Connection monitor error: {e}")
                time.sleep(1.0)

    def _go_offline(self):
        """Switch to offline mode."""
        if self._is_offline:
            return

        self._is_offline = True
        logger.warning("OFFLINE MODE: Server connection lost, switching to autonomous operation")

        # Initialize edge inference if available and not already running
        if self.edge_inference and not self.edge_inference.is_initialized:
            logger.info("Initializing edge inference for offline mode")
            self.edge_inference.initialize()

        if self._on_mode_change:
            self._on_mode_change(True)

    def _go_online(self):
        """Switch back to online mode."""
        if not self._is_offline:
            return

        self._is_offline = False
        logger.info("ONLINE MODE: Server connection restored")

        # Send queued alerts
        self._flush_alert_queue()

        if self._on_mode_change:
            self._on_mode_change(False)

    def process_frame_offline(self, frame, timestamp: float) -> Optional[list]:
        """
        Process a frame in offline mode.
        Returns detections if inference is available.
        """
        if not self._is_offline:
            return None

        if not self.edge_inference or not self.edge_inference.is_initialized:
            return None

        # Run local inference
        detections = self.edge_inference.run(frame, timestamp)

        if detections:
            self._handle_offline_detections(detections, timestamp)

        return detections

    def _handle_offline_detections(self, detections: list, timestamp: float):
        """Handle detections in offline mode."""
        if not self.config.auto_respond:
            return

        # Find highest threat level
        threat_levels = ["low", "medium", "high", "critical"]
        max_threat = "low"

        for det in detections:
            if hasattr(det, 'threat_level') and det.threat_level:
                level = det.threat_level.value if hasattr(det.threat_level, 'value') else str(det.threat_level)
                if level in threat_levels:
                    if threat_levels.index(level) > threat_levels.index(max_threat):
                        max_threat = level

        actions_taken = []

        # Respond based on threat level
        if max_threat in ["high", "critical"]:
            logger.warning(f"OFFLINE: {max_threat.upper()} threat detected, activating alarm")
            if self.gpio:
                self.gpio.activate_alarm()
                actions_taken.append("alarm_activated")

        if max_threat == "critical":
            logger.warning("OFFLINE: CRITICAL threat, activating sprayer")
            if self.gpio:
                if self.gpio.activate_sprayer():
                    actions_taken.append("sprayer_activated")

            # Use targeting if available
            if self.targeting and len(detections) > 0:
                self.targeting.process_detections(detections, timestamp=timestamp)

        # Queue alert for when we're back online
        self._queue_alert(detections, max_threat, actions_taken, timestamp)

    def _queue_alert(
        self,
        detections: list,
        threat_level: str,
        actions_taken: list,
        timestamp: float,
    ):
        """Queue an alert to send when back online."""
        if not self.config.queue_alerts:
            return

        alert = QueuedAlert(
            timestamp=timestamp,
            device_id=self.device_id,
            detections=[
                d.to_dict() if hasattr(d, 'to_dict') else {"class_name": str(d)}
                for d in detections
            ],
            threat_level=threat_level,
            actions_taken=actions_taken,
        )

        try:
            self._alert_queue.put_nowait(alert)
        except queue.Full:
            # Remove oldest and add new
            try:
                self._alert_queue.get_nowait()
                self._alert_queue.put_nowait(alert)
            except queue.Empty:
                pass

        # Also save to disk
        self._save_alert_to_disk(alert)

    def _save_alert_to_disk(self, alert: QueuedAlert):
        """Save alert to disk for persistence."""
        try:
            filename = f"alert_{int(alert.timestamp * 1000)}.json"
            filepath = Path(self.config.offline_storage_dir) / filename

            with open(filepath, "w") as f:
                json.dump(alert.to_dict(), f)

        except Exception as e:
            logger.error(f"Failed to save offline alert: {e}")

    def _flush_alert_queue(self):
        """Send all queued alerts (called when back online)."""
        sent = 0

        while not self._alert_queue.empty():
            try:
                alert = self._alert_queue.get_nowait()
                # These will be sent via the communicator
                logger.info(f"Sending queued alert from {datetime.fromtimestamp(alert.timestamp)}")
                sent += 1
            except queue.Empty:
                break

        # Also load and send any alerts saved to disk
        storage_path = Path(self.config.offline_storage_dir)
        for alert_file in storage_path.glob("alert_*.json"):
            try:
                with open(alert_file) as f:
                    alert_data = json.load(f)
                logger.info(f"Sending saved alert: {alert_file.name}")
                # Remove file after sending
                alert_file.unlink()
                sent += 1
            except Exception as e:
                logger.error(f"Failed to process saved alert {alert_file}: {e}")

        if sent > 0:
            logger.info(f"Flushed {sent} queued alerts")

    def get_queued_alerts(self) -> list[dict]:
        """Get all queued alerts without removing them."""
        alerts = []

        # In-memory queue
        temp_queue = queue.Queue()
        while not self._alert_queue.empty():
            try:
                alert = self._alert_queue.get_nowait()
                alerts.append(alert.to_dict())
                temp_queue.put(alert)
            except queue.Empty:
                break

        # Restore queue
        while not temp_queue.empty():
            try:
                self._alert_queue.put_nowait(temp_queue.get_nowait())
            except (queue.Full, queue.Empty):
                break

        # Disk alerts
        storage_path = Path(self.config.offline_storage_dir)
        for alert_file in storage_path.glob("alert_*.json"):
            try:
                with open(alert_file) as f:
                    alerts.append(json.load(f))
            except Exception:
                pass

        return alerts

    def set_mode_change_callback(self, callback: Callable[[bool], None]):
        """Set callback for mode changes. Callback receives True for offline, False for online."""
        self._on_mode_change = callback

    @property
    def is_offline(self) -> bool:
        return self._is_offline

    @property
    def queued_alert_count(self) -> int:
        return self._alert_queue.qsize()
