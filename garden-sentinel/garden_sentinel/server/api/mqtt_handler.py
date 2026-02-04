"""
MQTT handler for server-side communication with edge devices.
"""

import json
import logging
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)

from garden_sentinel.shared import EdgeMessage, MessageType, ServerCommand


class MQTTHandler:
    """
    MQTT handler for communication with edge devices.
    """

    def __init__(
        self,
        broker: str = "localhost",
        port: int = 1883,
        topic_prefix: str = "garden-sentinel",
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.broker = broker
        self.port = port
        self.topic_prefix = topic_prefix
        self.username = username
        self.password = password

        self._client = None
        self._connected = False
        self._message_callbacks: list[Callable[[EdgeMessage], None]] = []

    def start(self):
        """Start the MQTT handler."""
        try:
            import paho.mqtt.client as mqtt

            self._client = mqtt.Client(client_id="garden-sentinel-server")

            if self.username:
                self._client.username_pw_set(self.username, self.password)

            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect
            self._client.on_message = self._on_message

            self._client.connect_async(self.broker, self.port)
            self._client.loop_start()

            logger.info(f"MQTT handler connecting to {self.broker}:{self.port}")

        except ImportError:
            logger.warning("paho-mqtt not installed, MQTT disabled")
        except Exception as e:
            logger.error(f"Failed to start MQTT handler: {e}")

    def stop(self):
        """Stop the MQTT handler."""
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
        logger.info("MQTT handler stopped")

    def _on_connect(self, client, userdata, flags, rc):
        """Handle MQTT connection."""
        if rc == 0:
            self._connected = True
            logger.info("MQTT connected to broker")

            # Subscribe to device messages
            topics = [
                (f"{self.topic_prefix}/devices/+/frame", 1),
                (f"{self.topic_prefix}/devices/+/heartbeat", 0),
                (f"{self.topic_prefix}/devices/+/status", 1),
                (f"{self.topic_prefix}/devices/+/detection_result", 1),
            ]
            client.subscribe(topics)
            logger.info(f"Subscribed to {len(topics)} topics")
        else:
            logger.error(f"MQTT connection failed with code {rc}")

    def _on_disconnect(self, client, userdata, rc):
        """Handle MQTT disconnection."""
        self._connected = False
        if rc != 0:
            logger.warning(f"MQTT disconnected unexpectedly (rc={rc})")

    def _on_message(self, client, userdata, msg):
        """Handle incoming MQTT messages."""
        try:
            message = EdgeMessage.from_json(msg.payload.decode())

            # Notify callbacks
            for callback in self._message_callbacks:
                try:
                    callback(message)
                except Exception as e:
                    logger.error(f"Message callback error: {e}")

        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")

    def send_command(self, command: ServerCommand):
        """Send a command to a device."""
        if not self._client or not self._connected:
            logger.warning("MQTT not connected, cannot send command")
            return

        # Determine topic
        if command.target_device == "*":
            topic = f"{self.topic_prefix}/commands/broadcast"
        else:
            topic = f"{self.topic_prefix}/commands/{command.target_device}"

        try:
            self._client.publish(topic, command.to_json(), qos=1)
            logger.debug(f"Command sent to {command.target_device}: {command.command_type.value}")
        except Exception as e:
            logger.error(f"Failed to send command: {e}")

    def add_message_callback(self, callback: Callable[[EdgeMessage], None]):
        """Add a callback for incoming messages."""
        self._message_callbacks.append(callback)

    @property
    def is_connected(self) -> bool:
        return self._connected
