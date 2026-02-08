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
from garden_sentinel.server.alerts.push_notifications import (
    PushNotificationManager,
    create_push_manager_from_config,
    PushNotification,
    NotificationPriority,
)
from garden_sentinel.server.storage import StorageManager, StorageConfig
from garden_sentinel.server.api.server import create_app, broadcast_alert
from garden_sentinel.server.api.mqtt_handler import MQTTHandler
from garden_sentinel.server.coordination import CameraCoordinator, CameraRegistration
from garden_sentinel.server.analytics import PatternAnalyzer, VisitPrediction
from garden_sentinel.server.health import HealthAggregator
from garden_sentinel.server.spatial import (
    GardenMap,
    Zone,
    ZoneType,
    CameraPlacement,
    FlightPath,
    Point,
    Polygon,
    ZoneTracker,
    DeterrenceTracker,
    MapRenderer,
    MapAPIEndpoint,
    DroneTracker,
    GPSConverter,
    GPSCoordinate,
)
from garden_sentinel.shared.metrics import (
    MetricsRegistry,
    GardenSentinelMetricsCollector,
    PrometheusExporter,
)

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

        # Multi-camera coordinator
        coord_config = self.config.get("coordination", {})
        self.camera_coordinator = CameraCoordinator(
            correlation_threshold=coord_config.get("correlation_threshold", 0.5),
            target_timeout=coord_config.get("target_timeout", 10.0),
            min_confidence_to_engage=coord_config.get("min_confidence_to_engage", 0.6),
            engagement_cooldown=coord_config.get("engagement_cooldown", 30.0),
        )

        # Register cameras from config
        for cam_config in self.config.get("cameras", []):
            camera = CameraRegistration(
                device_id=cam_config["device_id"],
                name=cam_config.get("name", cam_config["device_id"]),
                position_x=cam_config.get("position_x", 0.0),
                position_y=cam_config.get("position_y", 0.0),
                position_z=cam_config.get("position_z", 2.0),
                heading=cam_config.get("heading", 0.0),
                fov_horizontal=cam_config.get("fov_horizontal", 62.0),
                fov_vertical=cam_config.get("fov_vertical", 49.0),
                has_sprayer=cam_config.get("has_sprayer", True),
                has_pan_tilt=cam_config.get("has_pan_tilt", True),
                sprayer_range=cam_config.get("sprayer_range", 5.0),
            )
            self.camera_coordinator.register_camera(camera)

        # Push notifications
        push_config = self.config.get("push_notifications", {})
        self.push_manager = create_push_manager_from_config(push_config) if push_config else None

        # Pattern analyzer
        analytics_config = self.config.get("analytics", {})
        storage_config = self.config.get("storage", {})
        db_path = Path(storage_config.get("db_path", "data/garden_sentinel.db"))
        self.pattern_analyzer = PatternAnalyzer(
            db_path=db_path.parent / "patterns.db",
            prediction_lookahead_hours=analytics_config.get("prediction_lookahead_hours", 2),
            min_visits_for_pattern=analytics_config.get("min_visits_for_pattern", 3),
        )

        # Metrics
        metrics_config = self.config.get("metrics", {})
        self.metrics_registry = MetricsRegistry()
        self.metrics_collector = GardenSentinelMetricsCollector(
            registry=self.metrics_registry,
            device_id="server",
        )

        # Prometheus exporter
        self.prometheus_exporter = None
        if metrics_config.get("prometheus_enabled", True):
            self.prometheus_exporter = PrometheusExporter(
                registry=self.metrics_registry,
                port=metrics_config.get("prometheus_port", 9090),
            )

        # Health aggregator
        health_config = self.config.get("health", {})
        self.health_aggregator = HealthAggregator(
            db_path=db_path.parent / "health.db",
            offline_threshold=health_config.get("offline_threshold", 120.0),
        )

        # Spatial configuration - garden map
        spatial_config = self.config.get("spatial", {})
        self.garden_map = self._create_garden_map(spatial_config)

        # Zone tracker for predator entry/exit detection
        self.zone_tracker = ZoneTracker(
            garden_map=self.garden_map,
            track_timeout=spatial_config.get("track_timeout", 60.0),
        )

        # Deterrence tracker
        protected_zone_ids = [
            z.zone_id for z in self.garden_map.zones
            if z.zone_type == ZoneType.PROTECTED
        ]
        self.deterrence_tracker = DeterrenceTracker(
            zone_tracker=self.zone_tracker,
            protected_zone_ids=protected_zone_ids,
            deterrence_window=spatial_config.get("deterrence_window", 30.0),
        )

        # Map renderer for visualization
        self.map_renderer = MapRenderer(
            garden_map=self.garden_map,
            zone_tracker=self.zone_tracker,
        )
        self.map_api = MapAPIEndpoint(self.garden_map, self.zone_tracker)

        # Drone tracker for mobile cameras
        gps_origin = spatial_config.get("gps_origin", {})
        self.gps_converter = None
        if gps_origin:
            self.gps_converter = GPSConverter(
                origin=GPSCoordinate(
                    latitude=gps_origin.get("latitude", 0.0),
                    longitude=gps_origin.get("longitude", 0.0),
                    altitude=gps_origin.get("altitude", 0.0),
                )
            )

        self.drone_tracker = DroneTracker(gps_converter=self.gps_converter)

        # FastAPI app
        self.app = create_app(
            detection_pipeline=self.detection_pipeline,
            alert_manager=self.alert_manager,
            storage_manager=self.storage_manager,
            mqtt_handler=self.mqtt_handler,
            camera_coordinator=self.camera_coordinator,
            pattern_analyzer=self.pattern_analyzer,
            health_aggregator=self.health_aggregator,
            map_api=self.map_api,
            drone_tracker=self.drone_tracker,
            metrics_collector=self.metrics_collector,
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

    def _create_garden_map(self, config: dict) -> GardenMap:
        """Create garden map from configuration."""
        # Parse boundary
        boundary = None
        if "boundary" in config:
            boundary_points = [Point(p[0], p[1]) for p in config["boundary"]]
            boundary = Polygon(boundary_points)

        # Parse zones
        zones = []
        for zone_config in config.get("zones", []):
            vertices = [Point(p[0], p[1]) for p in zone_config["vertices"]]
            zone = Zone(
                zone_id=zone_config["id"],
                name=zone_config["name"],
                zone_type=ZoneType(zone_config.get("type", "protected")),
                polygon=Polygon(vertices),
            )
            zones.append(zone)

        # Parse camera placements
        cameras = []
        for cam_config in config.get("cameras", []):
            camera = CameraPlacement(
                camera_id=cam_config["id"],
                name=cam_config.get("name", cam_config["id"]),
                position=Point(cam_config["x"], cam_config["y"]),
                heading=cam_config.get("heading", 0.0),
                fov=cam_config.get("fov", 90.0),
                range=cam_config.get("range", 15.0),
                altitude=cam_config.get("altitude"),
                is_mobile=cam_config.get("is_mobile", False),
            )
            cameras.append(camera)

        # Parse flight paths
        flight_paths = []
        for path_config in config.get("flight_paths", []):
            waypoints = [Point(p[0], p[1]) for p in path_config["waypoints"]]
            flight_path = FlightPath(
                path_id=path_config["id"],
                name=path_config.get("name", path_config["id"]),
                waypoints=waypoints,
                altitudes=path_config.get("altitudes", [20.0] * len(waypoints)),
                is_loop=path_config.get("is_loop", True),
            )
            flight_paths.append(flight_path)

        # GPS origin for coordinate conversion
        gps_origin = None
        if "gps_origin" in config:
            gps_origin = GPSCoordinate(
                latitude=config["gps_origin"]["latitude"],
                longitude=config["gps_origin"]["longitude"],
                altitude=config["gps_origin"].get("altitude", 0.0),
            )

        return GardenMap(
            name=config.get("name", "Garden"),
            boundary=boundary,
            zones=zones,
            cameras=cameras,
            flight_paths=flight_paths,
            gps_origin=gps_origin,
        )

    def _setup_callbacks(self):
        """Set up callbacks between components."""

        # Detection results -> Storage + Alerts + Coordinator + Pattern Analyzer + Zone Tracker
        def on_detection_result(result):
            # Track metrics
            self.metrics_collector.increment_frames_processed()

            # Save frame if there are detections
            if result.detections and result.annotated_frame is not None:
                self.storage_manager.save_frame(
                    result.device_id,
                    result.annotated_frame,
                    result.timestamp,
                    has_detections=True,
                )

            # Process through multi-camera coordinator
            for detection in result.detections:
                # Track detection metrics
                self.metrics_collector.increment_detections(
                    detection.predator_type if hasattr(detection, 'predator_type') else detection.class_name
                )

                asyncio.create_task(
                    self.camera_coordinator.process_detection(result.device_id, detection)
                )

                # Record in pattern analyzer
                asyncio.create_task(
                    self.pattern_analyzer.record_detection(
                        predator_type=detection.predator_type,
                        device_id=result.device_id,
                        confidence=detection.confidence,
                        bbox_x=detection.bbox.center_x if detection.bbox else None,
                        bbox_y=detection.bbox.center_y if detection.bbox else None,
                    )
                )

                # Update zone tracker if we have world position
                if hasattr(detection, 'world_position') and detection.world_position:
                    track_id = getattr(detection, 'track_id', f"{result.device_id}_{id(detection)}")
                    predator_type = getattr(detection, 'predator_type', detection.class_name)

                    zone_events = self.zone_tracker.update_position(
                        track_id=track_id,
                        predator_type=predator_type,
                        position=Point(detection.world_position[0], detection.world_position[1]),
                        timestamp=result.timestamp,
                    )

                    # Broadcast zone events to dashboard
                    for event in zone_events:
                        asyncio.create_task(broadcast_alert({
                            "type": "zone_event",
                            "event_type": event.event_type.value,
                            "zone_name": event.zone_name,
                            "predator_type": event.predator_type,
                            "track_id": event.track_id,
                        }))

        self.detection_pipeline.add_result_callback(on_detection_result)

        # Alerts -> Alert manager + Storage + Push notifications
        def on_alert(alert):
            # Get the latest frame for the alert
            frame = None  # Would need to cache frames
            self.alert_manager.handle_alert(alert, frame)
            self.storage_manager.save_event(alert, frame)

            # Broadcast to WebSocket clients
            asyncio.create_task(broadcast_alert(alert.to_dict()))

            # Send push notification
            if self.push_manager:
                asyncio.create_task(self.push_manager.send_alert(alert))

        self.detection_pipeline.add_alert_callback(on_alert)

        # Alert manager -> MQTT commands
        if self.mqtt_handler:
            self.alert_manager.set_command_callback(self.mqtt_handler.send_command)

        # Camera coordinator callbacks
        async def on_coordinator_engage(device_id: str, target):
            """Called when coordinator decides to engage a target."""
            logger.info(f"Coordinator engaging {target.predator_type} via {device_id}")
            if self.mqtt_handler:
                self.mqtt_handler.send_command(device_id, "spray", {
                    "target_id": target.target_id,
                    "duration": 3.0,
                })

            # Track spray in deterrence tracker
            self.deterrence_tracker.record_spray(target.target_id)

            # Track metrics
            self.metrics_collector.increment_sprays(target.predator_type)

            # Record spray event for pattern analysis
            await self.pattern_analyzer.record_spray_event(target.predator_type)

        async def on_coordinator_handoff(target, from_camera: str, to_camera: str):
            """Called when tracking hands off between cameras."""
            logger.info(f"Handoff: {target.target_id} from {from_camera} to {to_camera}")
            if self.mqtt_handler:
                # Tell old camera to stop tracking
                self.mqtt_handler.send_command(from_camera, "stop_tracking", {})
                # Tell new camera to start tracking
                self.mqtt_handler.send_command(to_camera, "track_target", {
                    "target_id": target.target_id,
                    "predator_type": target.predator_type,
                })

        async def on_target_lost(target):
            """Called when a target is lost."""
            logger.info(f"Target lost: {target.target_id} ({target.predator_type})")

            # Check each protected zone for deterrence
            for zone_id in self.deterrence_tracker.protected_zone_ids:
                if self.zone_tracker.check_deterred(target.target_id, zone_id):
                    # Target was successfully deterred
                    logger.info(f"Target {target.target_id} was deterred from zone {zone_id}")
                    self.metrics_collector.increment_deterred(target.predator_type)
                    await self.pattern_analyzer.record_deterrence(target.predator_type)
                    break

            # Clean up from zone tracker
            self.zone_tracker.cleanup_expired_tracks()

        self.camera_coordinator.set_callbacks(
            on_engage=on_coordinator_engage,
            on_handoff=on_coordinator_handoff,
            on_target_lost=on_target_lost,
        )

        # Pattern analyzer prediction callback
        async def on_prediction(prediction: VisitPrediction):
            """Called when pattern analyzer predicts a visit."""
            logger.info(f"Prediction: {prediction.message}")

            # Send push notification for predictions
            if self.push_manager and prediction.risk_score > 0.6:
                notification = PushNotification(
                    title=f"⚠️ {prediction.predator_type.title()} Expected",
                    message=prediction.message,
                    priority=NotificationPriority.NORMAL,
                    tags=[f"prediction:{prediction.predator_type}"],
                )
                await self.push_manager.send(notification)

            # Broadcast to dashboard
            await broadcast_alert({
                "type": "prediction",
                "predator_type": prediction.predator_type,
                "window": str(prediction.predicted_window),
                "risk_score": prediction.risk_score,
                "message": prediction.message,
            })

        self.pattern_analyzer.set_prediction_callback(on_prediction)

    async def start_async(self):
        """Start async components."""
        # Start camera coordinator
        await self.camera_coordinator.start()

        # Start pattern analyzer
        await self.pattern_analyzer.start()

        logger.info("Async components started")

    async def stop_async(self):
        """Stop async components."""
        await self.camera_coordinator.stop()
        await self.pattern_analyzer.stop()
        logger.info("Async components stopped")

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

        # Start Prometheus exporter
        if self.prometheus_exporter:
            self.prometheus_exporter.start()
            logger.info(f"Prometheus metrics available at http://localhost:{self.prometheus_exporter.port}/metrics")

        # Start health aggregator
        self.health_aggregator.start()

        # Set up callbacks
        self._setup_callbacks()

        # Start async components in background
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.start_async())

        logger.info("Garden Sentinel Server started")
        logger.info(f"  - Detection pipeline: active")
        logger.info(f"  - Cameras registered: {len(self.camera_coordinator.cameras)}")
        logger.info(f"  - Zones configured: {len(self.garden_map.zones)}")
        logger.info(f"  - Flight paths: {len(self.garden_map.flight_paths)}")
        return True

    def stop(self):
        """Stop all server components."""
        logger.info("Stopping Garden Sentinel Server")

        # Stop async components
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.stop_async())

        self.detection_pipeline.stop()

        if self.mqtt_handler:
            self.mqtt_handler.stop()

        if self.prometheus_exporter:
            self.prometheus_exporter.stop()

        self.health_aggregator.stop()

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
