#!/usr/bin/env python3
"""
Garden Sentinel Server - Main Application

Runs the central server for processing video from edge devices,
running detection, and coordinating alerts.
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

import uvicorn
import yaml

from garden_sentinel.server.detection import DetectionPipeline, DetectionConfig
from garden_sentinel.server.alerts import AlertManager, AlertConfig, NotificationConfig, ThreatLevelActions
from garden_sentinel.server.storage import StorageManager, StorageConfig
from garden_sentinel.server.api.server import create_app, broadcast_alert
from garden_sentinel.server.api.mqtt_handler import MQTTHandler

logger = logging.getLogger(__name__)


class GardenSentinelServer:
    """Main server application."""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)

        # Initialize components
        self.detection_pipeline = DetectionPipeline(
            config=DetectionConfig(**self.config.get("detection", {})),
            max_queue_size=self.config.get("detection", {}).get("max_queue_size", 100),
            num_workers=1,  # Single worker for GPU
        )

        self.storage_manager = StorageManager(
            config=StorageConfig(**self.config.get("storage", {}))
        )

        self.alert_manager = AlertManager(
            config=self._parse_alert_config(self.config.get("alerts", {}))
        )

        # MQTT handler
        mqtt_config = self.config.get("mqtt", {})
        self.mqtt_handler = MQTTHandler(
            broker=mqtt_config.get("broker", "localhost"),
            port=mqtt_config.get("port", 1883),
            topic_prefix=mqtt_config.get("topic_prefix", "garden-sentinel"),
            username=mqtt_config.get("username"),
            password=mqtt_config.get("password"),
        ) if mqtt_config.get("enabled", False) else None

        # FastAPI app
        self.app = create_app(
            detection_pipeline=self.detection_pipeline,
            alert_manager=self.alert_manager,
            storage_manager=self.storage_manager,
            mqtt_handler=self.mqtt_handler,
        )

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        config_file = Path(config_path)

        if not config_file.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}

        with open(config_file) as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {config_path}")
        return config

    def _parse_alert_config(self, config: dict) -> AlertConfig:
        """Parse alert configuration."""
        notif_config = config.get("notifications", {})

        notifications = NotificationConfig(
            pushover_enabled=notif_config.get("pushover", {}).get("enabled", False),
            pushover_user_key=notif_config.get("pushover", {}).get("user_key", ""),
            pushover_api_token=notif_config.get("pushover", {}).get("api_token", ""),
            email_enabled=notif_config.get("email", {}).get("enabled", False),
            email_smtp_host=notif_config.get("email", {}).get("smtp_host", ""),
            email_smtp_port=notif_config.get("email", {}).get("smtp_port", 587),
            email_username=notif_config.get("email", {}).get("username", ""),
            email_password=notif_config.get("email", {}).get("password", ""),
            email_from=notif_config.get("email", {}).get("from_address", ""),
            email_to=notif_config.get("email", {}).get("to_addresses", []),
            webhook_enabled=notif_config.get("webhook", {}).get("enabled", False),
            webhook_url=notif_config.get("webhook", {}).get("url", ""),
            home_assistant_enabled=notif_config.get("home_assistant", {}).get("enabled", False),
            home_assistant_url=notif_config.get("home_assistant", {}).get("url", ""),
            home_assistant_token=notif_config.get("home_assistant", {}).get("token", ""),
        )

        def parse_actions(level_config: dict) -> ThreatLevelActions:
            return ThreatLevelActions(
                log=level_config.get("log", True),
                notify=level_config.get("notify", False),
                alarm=level_config.get("alarm", False),
                sprayer=level_config.get("sprayer", False),
            )

        return AlertConfig(
            cooldown_s=config.get("cooldown_s", 30),
            low=parse_actions(config.get("low", {})),
            medium=parse_actions(config.get("medium", {})),
            high=parse_actions(config.get("high", {})),
            critical=parse_actions(config.get("critical", {})),
            notifications=notifications,
        )

    def _setup_callbacks(self):
        """Set up callbacks between components."""

        # Detection results -> Storage + Alerts
        def on_detection_result(result):
            # Save frame if there are detections
            if result.detections and result.annotated_frame is not None:
                self.storage_manager.save_frame(
                    result.device_id,
                    result.annotated_frame,
                    result.timestamp,
                    has_detections=True,
                )

        self.detection_pipeline.add_result_callback(on_detection_result)

        # Alerts -> Alert manager + Storage
        def on_alert(alert):
            # Get the latest frame for the alert
            frame = None  # Would need to cache frames
            self.alert_manager.handle_alert(alert, frame)
            self.storage_manager.save_event(alert, frame)

            # Broadcast to WebSocket clients
            asyncio.create_task(broadcast_alert(alert.to_dict()))

        self.detection_pipeline.add_alert_callback(on_alert)

        # Alert manager -> MQTT commands
        if self.mqtt_handler:
            self.alert_manager.set_command_callback(self.mqtt_handler.send_command)

    def start(self):
        """Start all server components."""
        logging.basicConfig(
            level=getattr(logging, self.config.get("logging", {}).get("level", "INFO")),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        logger.info("Starting Garden Sentinel Server")

        # Start detection pipeline
        if not self.detection_pipeline.start():
            logger.error("Failed to start detection pipeline")
            return False

        # Start MQTT handler
        if self.mqtt_handler:
            self.mqtt_handler.start()

        # Set up callbacks
        self._setup_callbacks()

        logger.info("Garden Sentinel Server started")
        return True

    def stop(self):
        """Stop all server components."""
        logger.info("Stopping Garden Sentinel Server")

        self.detection_pipeline.stop()

        if self.mqtt_handler:
            self.mqtt_handler.stop()

        self.storage_manager.close()

        logger.info("Garden Sentinel Server stopped")

    def run(self):
        """Run the server."""
        if not self.start():
            sys.exit(1)

        server_config = self.config.get("server", {})

        try:
            uvicorn.run(
                self.app,
                host=server_config.get("host", "0.0.0.0"),
                port=server_config.get("port", 5000),
                log_level="info",
            )
        finally:
            self.stop()


def main():
    parser = argparse.ArgumentParser(description="Garden Sentinel Server")
    parser.add_argument(
        "-c", "--config",
        default="config/config.yaml",
        help="Path to configuration file",
    )

    args = parser.parse_args()

    server = GardenSentinelServer(args.config)

    # Handle signals
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    server.run()


if __name__ == "__main__":
    main()
