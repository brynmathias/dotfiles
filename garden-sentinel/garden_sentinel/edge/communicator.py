"""
Communication module for edge device.
Handles MQTT messaging and HTTP communication with central server.
"""

import base64
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import requests

logger = logging.getLogger(__name__)

from garden_sentinel.shared import (
    CommandType,
    EdgeMessage,
    MessageType,
    ServerCommand,
)


@dataclass
class ServerConfig:
    host: str = "192.168.1.100"
    port: int = 5000
    mqtt_enabled: bool = True
    mqtt_broker: str = "192.168.1.100"
    mqtt_port: int = 1883
    mqtt_topic_prefix: str = "garden-sentinel"
    mqtt_username: Optional[str] = None
    mqtt_password: Optional[str] = None
    upload_frames: bool = True
    upload_interval_ms: int = 500
    heartbeat_interval_s: int = 30


class Communicator:
    """
    Handles communication between edge device and central server.
    Supports both MQTT for real-time messaging and HTTP for frame uploads.
    """

    def __init__(self, device_id: str, config: ServerConfig):
        self.device_id = device_id
        self.config = config

        self._mqtt_client = None
        self._running = False

        self._frame_queue: queue.Queue = queue.Queue(maxsize=10)
        self._upload_thread = None
        self._heartbeat_thread = None

        self._command_callbacks: list[Callable[[ServerCommand], None]] = []
        self._connected = False

        self._last_upload_time = 0

    def start(self):
        """Start communication services."""
        self._running = True

        # Initialize MQTT if enabled
        if self.config.mqtt_enabled:
            self._init_mqtt()

        # Start frame upload thread
        if self.config.upload_frames:
            self._upload_thread = threading.Thread(target=self._upload_loop, daemon=True)
            self._upload_thread.start()

        # Start heartbeat thread
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

        logger.info("Communicator started")

    def stop(self):
        """Stop communication services."""
        self._running = False

        if self._mqtt_client:
            try:
                self._mqtt_client.loop_stop()
                self._mqtt_client.disconnect()
            except Exception as e:
                logger.warning(f"Error stopping MQTT: {e}")

        logger.info("Communicator stopped")

    def _init_mqtt(self):
        """Initialize MQTT client."""
        try:
            import paho.mqtt.client as mqtt

            self._mqtt_client = mqtt.Client(client_id=f"edge-{self.device_id}")

            if self.config.mqtt_username:
                self._mqtt_client.username_pw_set(
                    self.config.mqtt_username, self.config.mqtt_password
                )

            self._mqtt_client.on_connect = self._on_mqtt_connect
            self._mqtt_client.on_disconnect = self._on_mqtt_disconnect
            self._mqtt_client.on_message = self._on_mqtt_message

            self._mqtt_client.connect_async(
                self.config.mqtt_broker, self.config.mqtt_port
            )
            self._mqtt_client.loop_start()

            logger.info(f"MQTT connecting to {self.config.mqtt_broker}:{self.config.mqtt_port}")

        except ImportError:
            logger.warning("paho-mqtt not installed, MQTT disabled")
        except Exception as e:
            logger.error(f"Failed to initialize MQTT: {e}")

    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """Handle MQTT connection."""
        if rc == 0:
            self._connected = True
            logger.info("MQTT connected")

            # Subscribe to command topic
            command_topic = f"{self.config.mqtt_topic_prefix}/commands/{self.device_id}"
            broadcast_topic = f"{self.config.mqtt_topic_prefix}/commands/broadcast"

            client.subscribe([(command_topic, 1), (broadcast_topic, 1)])
            logger.info(f"Subscribed to {command_topic} and {broadcast_topic}")
        else:
            logger.error(f"MQTT connection failed with code {rc}")

    def _on_mqtt_disconnect(self, client, userdata, rc):
        """Handle MQTT disconnection."""
        self._connected = False
        if rc != 0:
            logger.warning(f"MQTT disconnected unexpectedly (rc={rc})")

    def _on_mqtt_message(self, client, userdata, msg):
        """Handle incoming MQTT messages."""
        try:
            command = ServerCommand.from_json(msg.payload.decode())

            # Check if command is for this device
            if command.target_device not in [self.device_id, "*"]:
                return

            logger.info(f"Received command: {command.command_type.value}")

            # Notify callbacks
            for callback in self._command_callbacks:
                try:
                    callback(command)
                except Exception as e:
                    logger.error(f"Command callback error: {e}")

        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")

    def add_command_callback(self, callback: Callable[[ServerCommand], None]):
        """Add a callback for incoming commands."""
        self._command_callbacks.append(callback)

    def send_message(self, message_type: MessageType, payload: dict):
        """Send a message to the server via MQTT."""
        if not self._mqtt_client or not self._connected:
            return

        message = EdgeMessage(
            device_id=self.device_id,
            message_type=message_type,
            payload=payload,
        )

        topic = f"{self.config.mqtt_topic_prefix}/devices/{self.device_id}/{message_type.value}"

        try:
            self._mqtt_client.publish(topic, message.to_json(), qos=1)
        except Exception as e:
            logger.error(f"Failed to send MQTT message: {e}")

    def send_detection(self, detections: list, frame_jpeg: Optional[bytes] = None):
        """Send detection results to server."""
        payload = {
            "detections": [d.to_dict() if hasattr(d, 'to_dict') else d for d in detections],
        }

        if frame_jpeg:
            payload["frame_b64"] = base64.b64encode(frame_jpeg).decode()

        self.send_message(MessageType.DETECTION_RESULT, payload)

    def queue_frame(self, frame_jpeg: bytes, timestamp: float, force: bool = False):
        """Queue a frame for upload to server."""
        # Rate limiting
        if not force:
            time_since_last = (timestamp - self._last_upload_time) * 1000
            if time_since_last < self.config.upload_interval_ms:
                return

        self._last_upload_time = timestamp

        try:
            self._frame_queue.put_nowait((frame_jpeg, timestamp))
        except queue.Full:
            # Drop oldest frame
            try:
                self._frame_queue.get_nowait()
                self._frame_queue.put_nowait((frame_jpeg, timestamp))
            except queue.Empty:
                pass

    def _upload_loop(self):
        """Background thread for uploading frames via HTTP."""
        while self._running:
            try:
                frame_jpeg, timestamp = self._frame_queue.get(timeout=1.0)

                url = f"http://{self.config.host}:{self.config.port}/api/frames"

                response = requests.post(
                    url,
                    files={"frame": ("frame.jpg", frame_jpeg, "image/jpeg")},
                    data={
                        "device_id": self.device_id,
                        "timestamp": timestamp,
                    },
                    timeout=5.0,
                )

                if response.status_code != 200:
                    logger.warning(f"Frame upload failed: {response.status_code}")

            except queue.Empty:
                continue
            except requests.exceptions.RequestException as e:
                logger.warning(f"Frame upload error: {e}")
                time.sleep(1.0)
            except Exception as e:
                logger.error(f"Upload loop error: {e}")

    def _heartbeat_loop(self):
        """Background thread for sending heartbeats."""
        while self._running:
            try:
                self.send_message(MessageType.HEARTBEAT, {
                    "uptime": time.time(),
                    "connected": self._connected,
                })

                time.sleep(self.config.heartbeat_interval_s)

            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                time.sleep(5.0)

    def send_status(self, status: dict):
        """Send device status to server."""
        self.send_message(MessageType.STATUS, status)

    @property
    def is_connected(self) -> bool:
        """Check if connected to server."""
        return self._connected
