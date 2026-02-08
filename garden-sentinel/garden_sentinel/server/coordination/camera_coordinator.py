"""
Multi-camera coordination for tracking predators across camera views.

Handles:
- Detection correlation across cameras
- Target hand-off between cameras
- Engagement arbitration (which camera/sprayer responds)
- Triangulation for real-world position estimation
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable, Awaitable
from collections import defaultdict

from ...shared.protocol import Detection, BoundingBox, ThreatLevel

logger = logging.getLogger(__name__)


class EngagementStatus(Enum):
    """Status of camera engagement with a target."""
    IDLE = "idle"
    TRACKING = "tracking"
    ENGAGING = "engaging"
    COOLDOWN = "cooldown"


@dataclass
class CameraRegistration:
    """Registration info for a camera in the system."""
    device_id: str
    name: str
    # Position in garden coordinates (meters from origin)
    position_x: float = 0.0
    position_y: float = 0.0
    position_z: float = 2.0  # Height, default 2m
    # Camera orientation (degrees, 0 = North, clockwise)
    heading: float = 0.0
    # Field of view (degrees)
    fov_horizontal: float = 62.0
    fov_vertical: float = 49.0
    # Capabilities
    has_sprayer: bool = True
    has_pan_tilt: bool = True
    sprayer_range: float = 5.0  # meters
    # State
    status: EngagementStatus = EngagementStatus.IDLE
    current_target_id: Optional[str] = None
    last_detection_time: float = 0.0
    last_heartbeat: float = field(default_factory=time.time)

    @property
    def is_online(self) -> bool:
        """Check if camera is online (heartbeat within last 30s)."""
        return time.time() - self.last_heartbeat < 30.0


@dataclass
class TrackedTarget:
    """A target being tracked across multiple cameras."""
    target_id: str
    predator_type: str
    threat_level: ThreatLevel
    # World position estimate (if triangulated)
    world_x: Optional[float] = None
    world_y: Optional[float] = None
    # Velocity estimate (m/s)
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    # Which cameras see this target
    camera_detections: dict[str, Detection] = field(default_factory=dict)
    # Tracking state
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    # Engagement
    assigned_camera: Optional[str] = None
    engagement_started: Optional[float] = None
    times_sprayed: int = 0

    @property
    def age(self) -> float:
        """How long this target has been tracked (seconds)."""
        return time.time() - self.first_seen

    @property
    def time_since_seen(self) -> float:
        """Time since last detection (seconds)."""
        return time.time() - self.last_seen

    @property
    def camera_count(self) -> int:
        """Number of cameras currently seeing this target."""
        # Only count recent detections (within 2 seconds)
        cutoff = time.time() - 2.0
        return sum(
            1 for d in self.camera_detections.values()
            if d.timestamp and d.timestamp > cutoff
        )


class CameraCoordinator:
    """
    Coordinates multiple cameras tracking predators.

    Features:
    - Correlates detections across cameras (same predator = same target)
    - Estimates world position via triangulation
    - Hands off tracking as predator moves between camera views
    - Arbitrates which camera should engage (closest, best angle, etc.)
    """

    def __init__(
        self,
        correlation_threshold: float = 0.5,  # Max seconds between detections to correlate
        target_timeout: float = 10.0,  # Seconds without detection before target lost
        min_confidence_to_engage: float = 0.6,
        engagement_cooldown: float = 30.0,  # Seconds between engagements per target
    ):
        self.correlation_threshold = correlation_threshold
        self.target_timeout = target_timeout
        self.min_confidence_to_engage = min_confidence_to_engage
        self.engagement_cooldown = engagement_cooldown

        # Registered cameras
        self._cameras: dict[str, CameraRegistration] = {}

        # Active targets being tracked
        self._targets: dict[str, TrackedTarget] = {}

        # Target ID counter
        self._next_target_id = 1

        # Callbacks
        self._on_new_target: Optional[Callable[[TrackedTarget], Awaitable[None]]] = None
        self._on_target_lost: Optional[Callable[[TrackedTarget], Awaitable[None]]] = None
        self._on_engage: Optional[Callable[[str, TrackedTarget], Awaitable[None]]] = None
        self._on_handoff: Optional[Callable[[TrackedTarget, str, str], Awaitable[None]]] = None

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Background task
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the coordinator background tasks."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Camera coordinator started")

    async def stop(self):
        """Stop the coordinator."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Camera coordinator stopped")

    def register_camera(self, camera: CameraRegistration):
        """Register a camera with the coordinator."""
        self._cameras[camera.device_id] = camera
        logger.info(f"Registered camera: {camera.device_id} ({camera.name}) at ({camera.position_x}, {camera.position_y})")

    def unregister_camera(self, device_id: str):
        """Unregister a camera."""
        if device_id in self._cameras:
            del self._cameras[device_id]
            logger.info(f"Unregistered camera: {device_id}")

    def update_heartbeat(self, device_id: str):
        """Update camera heartbeat timestamp."""
        if device_id in self._cameras:
            self._cameras[device_id].last_heartbeat = time.time()

    def set_callbacks(
        self,
        on_new_target: Optional[Callable[[TrackedTarget], Awaitable[None]]] = None,
        on_target_lost: Optional[Callable[[TrackedTarget], Awaitable[None]]] = None,
        on_engage: Optional[Callable[[str, TrackedTarget], Awaitable[None]]] = None,
        on_handoff: Optional[Callable[[TrackedTarget, str, str], Awaitable[None]]] = None,
    ):
        """Set event callbacks."""
        self._on_new_target = on_new_target
        self._on_target_lost = on_target_lost
        self._on_engage = on_engage
        self._on_handoff = on_handoff

    async def process_detection(
        self,
        device_id: str,
        detection: Detection,
    ) -> Optional[TrackedTarget]:
        """
        Process a detection from a camera.

        Returns the TrackedTarget this detection was associated with.
        """
        async with self._lock:
            if device_id not in self._cameras:
                logger.warning(f"Detection from unregistered camera: {device_id}")
                return None

            camera = self._cameras[device_id]
            camera.last_detection_time = time.time()

            # Try to correlate with existing target
            target = self._correlate_detection(device_id, detection)

            if target is None:
                # New target
                target = self._create_target(device_id, detection)
                logger.info(
                    f"New target {target.target_id}: {target.predator_type} "
                    f"(threat: {target.threat_level.value}) from {device_id}"
                )
                if self._on_new_target:
                    asyncio.create_task(self._on_new_target(target))
            else:
                # Update existing target
                self._update_target(target, device_id, detection)

            # Check if we should engage
            await self._evaluate_engagement(target)

            return target

    def _correlate_detection(
        self,
        device_id: str,
        detection: Detection,
    ) -> Optional[TrackedTarget]:
        """
        Try to correlate a detection with an existing target.

        Uses:
        - Same predator type
        - Temporal proximity (recent detections)
        - Spatial proximity (if world position known)
        """
        now = time.time()
        best_match: Optional[TrackedTarget] = None
        best_score = 0.0

        for target in self._targets.values():
            # Must be same predator type
            if target.predator_type != detection.predator_type:
                continue

            # Must have been seen recently
            if target.time_since_seen > self.target_timeout:
                continue

            # Calculate correlation score
            score = 0.0

            # Temporal score: higher if seen recently
            time_factor = max(0, 1.0 - target.time_since_seen / self.correlation_threshold)
            score += time_factor * 0.5

            # If this camera already sees the target, high correlation
            if device_id in target.camera_detections:
                prev_detection = target.camera_detections[device_id]
                # Check bounding box overlap
                if prev_detection.bbox and detection.bbox:
                    iou = self._calculate_iou(prev_detection.bbox, detection.bbox)
                    score += iou * 0.5

            # If other cameras see it, moderate correlation
            elif target.camera_count > 0:
                score += 0.3

            if score > best_score and score > 0.3:
                best_score = score
                best_match = target

        return best_match

    def _create_target(self, device_id: str, detection: Detection) -> TrackedTarget:
        """Create a new tracked target from a detection."""
        target_id = f"T{self._next_target_id:04d}"
        self._next_target_id += 1

        target = TrackedTarget(
            target_id=target_id,
            predator_type=detection.predator_type,
            threat_level=detection.threat_level,
            camera_detections={device_id: detection},
        )

        self._targets[target_id] = target
        return target

    def _update_target(
        self,
        target: TrackedTarget,
        device_id: str,
        detection: Detection,
    ):
        """Update a target with a new detection."""
        # Store previous position for velocity calculation
        prev_detection = target.camera_detections.get(device_id)

        # Update detection
        target.camera_detections[device_id] = detection
        target.last_seen = time.time()

        # Update threat level if higher
        if detection.threat_level.value > target.threat_level.value:
            target.threat_level = detection.threat_level

        # Calculate velocity if we have previous detection from same camera
        if prev_detection and prev_detection.bbox and detection.bbox:
            dt = (detection.timestamp or time.time()) - (prev_detection.timestamp or 0)
            if dt > 0 and dt < 2.0:  # Reasonable time delta
                # Velocity in normalized coordinates per second
                dx = detection.bbox.center_x - prev_detection.bbox.center_x
                dy = detection.bbox.center_y - prev_detection.bbox.center_y
                # Smooth velocity update
                alpha = 0.3
                target.velocity_x = alpha * (dx / dt) + (1 - alpha) * target.velocity_x
                target.velocity_y = alpha * (dy / dt) + (1 - alpha) * target.velocity_y

    async def _evaluate_engagement(self, target: TrackedTarget):
        """Evaluate whether to engage a target and which camera should do it."""
        # Check if target warrants engagement
        if target.threat_level == ThreatLevel.LOW:
            return

        # Check cooldown
        if target.engagement_started:
            if time.time() - target.engagement_started < self.engagement_cooldown:
                return

        # Find best camera to engage
        best_camera: Optional[str] = None
        best_score = 0.0

        for device_id, camera in self._cameras.items():
            if not camera.is_online:
                continue
            if not camera.has_sprayer:
                continue
            if camera.status == EngagementStatus.ENGAGING:
                continue
            if camera.status == EngagementStatus.COOLDOWN:
                continue

            # Does this camera see the target?
            if device_id not in target.camera_detections:
                continue

            detection = target.camera_detections[device_id]

            # Check confidence
            if detection.confidence < self.min_confidence_to_engage:
                continue

            # Score this camera
            score = 0.0

            # Higher confidence = better
            score += detection.confidence * 0.4

            # Currently assigned camera gets bonus
            if target.assigned_camera == device_id:
                score += 0.3

            # Camera already tracking = better (smoother engagement)
            if camera.status == EngagementStatus.TRACKING:
                score += 0.2

            # Prefer camera with target more centered
            if detection.bbox:
                center_dist = abs(detection.bbox.center_x - 0.5) + abs(detection.bbox.center_y - 0.5)
                score += (1.0 - center_dist) * 0.1

            if score > best_score:
                best_score = score
                best_camera = device_id

        if best_camera is None:
            return

        # Check for handoff
        if target.assigned_camera and target.assigned_camera != best_camera:
            old_camera = target.assigned_camera
            logger.info(f"Handing off target {target.target_id} from {old_camera} to {best_camera}")

            # Update old camera status
            if old_camera in self._cameras:
                self._cameras[old_camera].status = EngagementStatus.IDLE
                self._cameras[old_camera].current_target_id = None

            if self._on_handoff:
                asyncio.create_task(self._on_handoff(target, old_camera, best_camera))

        # Assign camera
        target.assigned_camera = best_camera
        camera = self._cameras[best_camera]
        camera.current_target_id = target.target_id

        # Engage if threat level high enough
        if target.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            if camera.status != EngagementStatus.ENGAGING:
                camera.status = EngagementStatus.ENGAGING
                target.engagement_started = time.time()
                target.times_sprayed += 1

                logger.info(
                    f"Camera {best_camera} engaging target {target.target_id} "
                    f"(spray #{target.times_sprayed})"
                )

                if self._on_engage:
                    asyncio.create_task(self._on_engage(best_camera, target))
        else:
            camera.status = EngagementStatus.TRACKING

    async def _cleanup_loop(self):
        """Background loop to clean up stale targets."""
        while True:
            try:
                await asyncio.sleep(1.0)
                await self._cleanup_stale_targets()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    async def _cleanup_stale_targets(self):
        """Remove targets that haven't been seen recently."""
        async with self._lock:
            stale_targets = [
                target for target in self._targets.values()
                if target.time_since_seen > self.target_timeout
            ]

            for target in stale_targets:
                logger.info(
                    f"Target {target.target_id} lost after {target.age:.1f}s "
                    f"(sprayed {target.times_sprayed} times)"
                )

                # Clear camera assignments
                if target.assigned_camera and target.assigned_camera in self._cameras:
                    camera = self._cameras[target.assigned_camera]
                    camera.status = EngagementStatus.IDLE
                    camera.current_target_id = None

                del self._targets[target.target_id]

                if self._on_target_lost:
                    asyncio.create_task(self._on_target_lost(target))

    def _calculate_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate Intersection over Union of two bounding boxes."""
        # Calculate intersection
        x1 = max(box1.x, box2.x)
        y1 = max(box1.y, box2.y)
        x2 = min(box1.x + box1.width, box2.x + box2.width)
        y2 = min(box1.y + box1.height, box2.y + box2.height)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)

        # Calculate union
        area1 = box1.width * box1.height
        area2 = box2.width * box2.height
        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    def get_active_targets(self) -> list[TrackedTarget]:
        """Get all currently active targets."""
        return list(self._targets.values())

    def get_camera_status(self) -> dict[str, dict]:
        """Get status of all cameras."""
        return {
            device_id: {
                "name": camera.name,
                "online": camera.is_online,
                "status": camera.status.value,
                "current_target": camera.current_target_id,
                "position": (camera.position_x, camera.position_y),
            }
            for device_id, camera in self._cameras.items()
        }

    def get_target(self, target_id: str) -> Optional[TrackedTarget]:
        """Get a specific target by ID."""
        return self._targets.get(target_id)
