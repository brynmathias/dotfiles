"""Garden Sentinel Edge Device Module"""

from garden_sentinel.edge.camera import Camera, CameraConfig
from garden_sentinel.edge.gpio_controller import GPIOController, GPIOConfig
from garden_sentinel.edge.motion_detector import MotionDetector, MotionConfig
from garden_sentinel.edge.communicator import Communicator, ServerConfig
from garden_sentinel.edge.edge_inference import EdgeInference, InferenceConfig
from garden_sentinel.edge.streaming import StreamingServer
from garden_sentinel.edge.tracker import ObjectTracker, TrackingConfig
from garden_sentinel.edge.servo_controller import ServoController, ServoConfig

__all__ = [
    "Camera", "CameraConfig",
    "GPIOController", "GPIOConfig",
    "MotionDetector", "MotionConfig",
    "Communicator", "ServerConfig",
    "EdgeInference", "InferenceConfig",
    "StreamingServer",
    "ObjectTracker", "TrackingConfig",
    "ServoController", "ServoConfig",
]
