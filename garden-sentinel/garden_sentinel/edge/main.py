#!/usr/bin/env python3
"""
Garden Sentinel Edge Device - Main Application

Runs on Raspberry Pi 5 to capture video, detect motion,
and communicate with the central server.
"""

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

import yaml

from garden_sentinel.shared import CommandType, ServerCommand, ThreatLevel

from garden_sentinel.edge.camera import Camera, CameraConfig
from garden_sentinel.edge.communicator import Communicator, ServerConfig
from garden_sentinel.edge.edge_inference import EdgeInference, InferenceConfig
from garden_sentinel.edge.gpio_controller import GPIOController, GPIOConfig
from garden_sentinel.edge.motion_detector import MotionDetector, MotionConfig
from garden_sentinel.edge.streaming import StreamingServer
from garden_sentinel.edge.health_monitor import HealthMonitor, INA219Monitor
from garden_sentinel.edge.targeting import TargetingSystem, TargetingConfig
from garden_sentinel.edge.offline_mode import OfflineManager, OfflineConfig
from garden_sentinel.edge.recorder import EventRecorder, RecorderConfig
from garden_sentinel.shared.metrics import (
    MetricsRegistry,
    GardenSentinelMetricsCollector,
    PrometheusExporter,
)

logger = logging.getLogger(__name__)


class GardenSentinelEdge:
    """
    Main application class for the Garden Sentinel edge device.
    """

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self._running = False

        # Initialize components
        self.camera = Camera(CameraConfig(**self.config.get("camera", {})))

        self.gpio = GPIOController(GPIOConfig(**self.config.get("gpio", {})))

        self.motion_detector = MotionDetector(
            MotionConfig(**self.config.get("motion_detection", {}))
        )

        self.communicator = Communicator(
            device_id=self.config["device"]["id"],
            config=ServerConfig(**self.config.get("server", {})),
        )

        self.edge_inference = EdgeInference(
            InferenceConfig(**self.config.get("edge_inference", {}))
        )

        # Streaming server
        streaming_config = self.config.get("streaming", {})
        self.streaming_server = StreamingServer(
            camera=self.camera,
            port=streaming_config.get("http_port", 8080),
            stream_path=streaming_config.get("http_path", "/stream"),
            jpeg_quality=streaming_config.get("jpeg_quality", 85),
            stream_fps=streaming_config.get("stream_fps", 15),
        ) if streaming_config.get("http_enabled", True) else None

        # Metrics collection
        metrics_config = self.config.get("metrics", {})
        self.metrics_registry = MetricsRegistry()
        self.metrics_collector = GardenSentinelMetricsCollector(
            registry=self.metrics_registry,
            device_id=self.config["device"]["id"],
        )

        # Prometheus exporter
        self.prometheus_exporter = None
        if metrics_config.get("prometheus_enabled", False):
            self.prometheus_exporter = PrometheusExporter(
                registry=self.metrics_registry,
                port=metrics_config.get("prometheus_port", 9090),
            )

        # Health monitoring
        health_config = self.config.get("health", {})
        battery_monitor = None
        if health_config.get("battery_monitor") == "ina219":
            try:
                battery_monitor = INA219Monitor(
                    i2c_address=health_config.get("i2c_address", 0x40),
                    battery_capacity_ah=health_config.get("battery_capacity_ah", 10.0),
                )
            except Exception as e:
                logger.warning(f"Could not initialize INA219: {e}")

        self.health_monitor = HealthMonitor(
            device_id=self.config["device"]["id"],
            battery_monitor=battery_monitor,
            server_url=f"http://{self.config.get('server', {}).get('host', 'localhost')}:"
                       f"{self.config.get('server', {}).get('port', 5000)}",
            check_interval=health_config.get("check_interval", 30.0),
            metrics_collector=self.metrics_collector,
        )

        # Targeting system for smart deterrence
        targeting_config = self.config.get("targeting", {})
        self.targeting = TargetingSystem(
            config=TargetingConfig(**targeting_config),
            gpio_controller=self.gpio,
        ) if targeting_config.get("enabled", True) else None

        # Offline mode manager
        offline_config = self.config.get("offline", {})
        self.offline_manager = OfflineManager(
            config=OfflineConfig(**offline_config),
        )

        # Event recorder
        recorder_config = self.config.get("recorder", {})
        self.recorder = EventRecorder(
            config=RecorderConfig(**recorder_config),
        ) if recorder_config.get("enabled", True) else None

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        config_file = Path(config_path)

        if not config_file.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return self._default_config()

        with open(config_file) as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {config_path}")
        return config

    def _default_config(self) -> dict:
        """Return default configuration."""
        return {
            "device": {"id": "garden-cam-01", "name": "Default Camera"},
            "camera": {"width": 1920, "height": 1080, "fps": 30},
            "streaming": {"http_enabled": True, "http_port": 8080},
            "server": {"host": "localhost", "port": 5000},
            "gpio": {},
            "motion_detection": {"enabled": True},
            "edge_inference": {"enabled": False},
        }

    def _setup_logging(self):
        """Configure logging based on config."""
        log_config = self.config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO").upper())

        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )

        # Add file handler if configured
        log_file = log_config.get("file")
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            from logging.handlers import RotatingFileHandler

            handler = RotatingFileHandler(
                log_file,
                maxBytes=log_config.get("max_size_mb", 50) * 1024 * 1024,
                backupCount=log_config.get("backup_count", 5),
            )
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            logging.getLogger().addHandler(handler)

    def _handle_command(self, command: ServerCommand):
        """Handle incoming commands from the server."""
        logger.info(f"Processing command: {command.command_type.value}")

        if command.command_type == CommandType.ACTIVATE_ALARM:
            duration = command.parameters.get("duration_s")
            self.gpio.activate_alarm(duration)

        elif command.command_type == CommandType.DEACTIVATE_ALARM:
            self.gpio.deactivate_alarm()

        elif command.command_type == CommandType.ACTIVATE_SPRAYER:
            duration = command.parameters.get("duration_s")
            self.gpio.activate_sprayer(duration)

        elif command.command_type == CommandType.DEACTIVATE_SPRAYER:
            self.gpio.deactivate_sprayer()

        elif command.command_type == CommandType.CAPTURE_SNAPSHOT:
            jpeg = self.camera.get_jpeg(quality=95)
            if jpeg:
                self.communicator.queue_frame(jpeg, time.time(), force=True)

        elif command.command_type == CommandType.UPDATE_CONFIG:
            # Handle dynamic config updates
            new_config = command.parameters.get("config", {})
            self._apply_config_update(new_config)

        elif command.command_type == CommandType.REBOOT:
            logger.warning("Reboot requested, shutting down...")
            self.stop()
            import os
            os.system("sudo reboot")

    def _apply_config_update(self, new_config: dict):
        """Apply configuration updates dynamically."""
        if "motion_detection" in new_config:
            md_config = new_config["motion_detection"]
            if "sensitivity" in md_config:
                self.motion_detector.config.sensitivity = md_config["sensitivity"]
            if "enabled" in md_config:
                self.motion_detector.config.enabled = md_config["enabled"]
            logger.info("Updated motion detection config")

        if "edge_inference" in new_config:
            ei_config = new_config["edge_inference"]
            if "confidence_threshold" in ei_config:
                self.edge_inference.config.confidence_threshold = ei_config["confidence_threshold"]
            logger.info("Updated edge inference config")

    def _on_frame(self, frame, timestamp: float):
        """Process each camera frame."""
        # Track frame for metrics
        self.metrics_collector.increment_frames_processed()

        # Check if we're in offline mode
        is_offline = self.offline_manager.is_offline

        # Check for motion
        motion_detected, contours = self.motion_detector.process_frame(frame, timestamp)

        # Run edge inference if enabled and motion detected
        detections = None
        if motion_detected or not self.motion_detector.config.enabled:
            detections = self.edge_inference.run(frame, timestamp)

        # Send detections to server (or queue if offline)
        if detections:
            # Record metrics
            for detection in detections:
                self.metrics_collector.increment_detections(detection.class_name)

            jpeg = self.camera.get_jpeg(quality=85)

            if is_offline:
                # Store locally for later sync
                self.offline_manager.queue_detection(detections, jpeg, timestamp)
            else:
                self.communicator.send_detection(detections, jpeg)

            # Record event if recorder is active
            if self.recorder:
                self.recorder.record_detection(frame, detections, timestamp)

            # Check threat level and respond locally
            for detection in detections:
                if detection.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                    logger.warning(
                        f"High threat detected: {detection.class_name} "
                        f"(confidence: {detection.confidence:.2f})"
                    )

                    # Use smart targeting if available
                    if self.targeting and detection.bbox:
                        spray_result = self.targeting.engage_target(
                            detection, frame.shape[:2]
                        )
                        if spray_result:
                            self.metrics_collector.increment_sprays(detection.class_name)
                    elif detection.threat_level == ThreatLevel.CRITICAL:
                        # Fallback: activate spray directly
                        self.gpio.activate_sprayer(duration=2.0)
                        self.gpio.activate_alarm()
                        self.gpio.blink_status_led(on_time=0.2, off_time=0.2, n=10)
                        self.metrics_collector.increment_sprays(detection.class_name)

        # Upload frames to server (if online)
        if not is_offline and (motion_detected or detections):
            jpeg = self.camera.get_jpeg()
            if jpeg:
                self.communicator.queue_frame(jpeg, timestamp, force=bool(detections))

    def start(self):
        """Start all components."""
        self._setup_logging()
        logger.info("Starting Garden Sentinel Edge Device")

        self._running = True

        # Start camera
        self.camera.start()
        self.camera.add_frame_callback(self._on_frame)

        # Start streaming server
        if self.streaming_server:
            self.streaming_server.start()

        # Initialize edge inference
        self.edge_inference.initialize()

        # Start communicator
        self.communicator.start()
        self.communicator.add_command_callback(self._handle_command)

        # Start health monitor
        self.health_monitor.start()

        # Start Prometheus exporter
        if self.prometheus_exporter:
            self.prometheus_exporter.start()

        # Start offline manager
        self.offline_manager.start()

        # Start event recorder
        if self.recorder:
            self.recorder.start()

        # Status LED on
        self.gpio.set_status_led(True)
        self.gpio.blink_status_led(on_time=0.1, off_time=0.1, n=3)

        # Send initial status
        self._send_status()

        logger.info(f"Garden Sentinel Edge Device '{self.config['device']['id']}' started")
        logger.info(f"  - Camera: {self.camera.config.width}x{self.camera.config.height}")
        logger.info(f"  - Stream: http://localhost:{self.config.get('streaming', {}).get('http_port', 8080)}")
        logger.info(f"  - Edge inference: {'enabled' if self.edge_inference.is_initialized else 'disabled'}")
        logger.info(f"  - GPIO: {'mock mode' if self.gpio.is_mock else 'active'}")
        logger.info(f"  - Health monitor: active (interval: {self.health_monitor.check_interval}s)")
        if self.prometheus_exporter:
            logger.info(f"  - Prometheus: http://localhost:{self.prometheus_exporter.port}/metrics")

    def _send_status(self):
        """Send device status to server."""
        status = {
            "device_id": self.config["device"]["id"],
            "device_name": self.config["device"].get("name", "Unknown"),
            "location": self.config["device"].get("location", "Unknown"),
            "camera_fps": self.camera.fps,
            "gpio_states": self.gpio.get_states(),
            "edge_inference_enabled": self.edge_inference.is_initialized,
            "motion_detection_enabled": self.motion_detector.config.enabled,
        }
        self.communicator.send_status(status)

    def run(self):
        """Run the main loop."""
        status_interval = 60  # Send status every 60 seconds
        last_status_time = time.time()

        try:
            while self._running:
                # Periodic status update
                if time.time() - last_status_time >= status_interval:
                    self._send_status()
                    last_status_time = time.time()

                time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()

    def stop(self):
        """Stop all components."""
        logger.info("Stopping Garden Sentinel Edge Device")
        self._running = False

        # Status LED off
        self.gpio.set_status_led(False)

        # Stop components
        self.camera.stop()
        if self.streaming_server:
            self.streaming_server.stop()
        self.communicator.stop()
        self.edge_inference.cleanup()
        self.gpio.cleanup()

        # Stop new components
        self.health_monitor.stop()
        if self.prometheus_exporter:
            self.prometheus_exporter.stop()
        self.offline_manager.stop()
        if self.recorder:
            self.recorder.stop()

        logger.info("Garden Sentinel Edge Device stopped")


def main():
    parser = argparse.ArgumentParser(description="Garden Sentinel Edge Device")
    parser.add_argument(
        "-c", "--config",
        default="/etc/garden-sentinel/edge.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle signals
    app = GardenSentinelEdge(args.config)

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        app.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    app.start()
    app.run()


if __name__ == "__main__":
    main()
