"""Garden Sentinel Server Module"""

from garden_sentinel.server.detection import PredatorDetector, DetectionConfig, DetectionPipeline
from garden_sentinel.server.alerts import (
    AlertManager,
    AlertConfig,
    PushNotificationManager,
    create_push_manager_from_config,
)
from garden_sentinel.server.storage import StorageManager, StorageConfig
from garden_sentinel.server.api import create_app
from garden_sentinel.server.coordination import (
    CameraCoordinator,
    CameraRegistration,
    TrackedTarget,
    Triangulator,
)
from garden_sentinel.server.analytics import (
    PatternAnalyzer,
    PredatorPattern,
    VisitPrediction,
)

__all__ = [
    "PredatorDetector", "DetectionConfig", "DetectionPipeline",
    "AlertManager", "AlertConfig",
    "PushNotificationManager", "create_push_manager_from_config",
    "StorageManager", "StorageConfig",
    "create_app",
    "CameraCoordinator", "CameraRegistration", "TrackedTarget", "Triangulator",
    "PatternAnalyzer", "PredatorPattern", "VisitPrediction",
]
