"""Garden Sentinel Edge Device Module"""

from garden_sentinel.edge.camera import Camera, CameraConfig
from garden_sentinel.edge.gpio_controller import GPIOController, GPIOConfig
from garden_sentinel.edge.motion_detector import MotionDetector, MotionConfig
from garden_sentinel.edge.communicator import Communicator, ServerConfig
from garden_sentinel.edge.edge_inference import EdgeInference, InferenceConfig
from garden_sentinel.edge.streaming import StreamingServer
from garden_sentinel.edge.tracker import ObjectTracker, TrackingConfig
from garden_sentinel.edge.servo_controller import ServoController, ServoConfig
from garden_sentinel.edge.targeting import TargetingController, TargetingConfig
from garden_sentinel.edge.offline_mode import OfflineModeHandler, OfflineConfig
from garden_sentinel.edge.recorder import EventRecorder, RecorderConfig
from garden_sentinel.edge.health_monitor import (
    HealthMonitor,
    HealthReport,
    HealthStatus,
    BatteryStatus,
    SystemStatus,
    NetworkStatus,
    CameraStatus,
    INA219Monitor,
    ADS1115Monitor,
    create_battery_monitor,
)

__all__ = [
    "Camera", "CameraConfig",
    "GPIOController", "GPIOConfig",
    "MotionDetector", "MotionConfig",
    "Communicator", "ServerConfig",
    "EdgeInference", "InferenceConfig",
    "StreamingServer",
    "ObjectTracker", "TrackingConfig",
    "ServoController", "ServoConfig",
    "TargetingController", "TargetingConfig",
    "OfflineModeHandler", "OfflineConfig",
    "EventRecorder", "RecorderConfig",
    "HealthMonitor", "HealthReport", "HealthStatus",
    "BatteryStatus", "SystemStatus", "NetworkStatus", "CameraStatus",
    "INA219Monitor", "ADS1115Monitor", "create_battery_monitor",
]
