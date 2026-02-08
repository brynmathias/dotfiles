"""
Triangulation for estimating world position from multiple camera views.

Uses camera positions and detection bounding boxes to estimate
where a predator actually is in the garden.
"""

import math
import logging
from dataclasses import dataclass
from typing import Optional

from .camera_coordinator import CameraRegistration
from ...shared.protocol import Detection, BoundingBox

logger = logging.getLogger(__name__)


@dataclass
class WorldPosition:
    """Estimated position in world coordinates."""
    x: float  # meters from origin
    y: float  # meters from origin
    confidence: float  # 0-1, based on triangulation quality
    timestamp: float


class Triangulator:
    """
    Estimates world position of targets from multiple camera views.

    Uses ray intersection from camera positions through detection centroids.
    """

    def __init__(self, ground_plane_z: float = 0.0):
        """
        Initialize triangulator.

        Args:
            ground_plane_z: Height of ground plane in meters (predators walk on this)
        """
        self.ground_plane_z = ground_plane_z

    def estimate_position(
        self,
        cameras: dict[str, CameraRegistration],
        detections: dict[str, Detection],
        timestamp: float,
    ) -> Optional[WorldPosition]:
        """
        Estimate world position from multiple camera detections.

        Args:
            cameras: Camera registrations keyed by device_id
            detections: Detections keyed by device_id

        Returns:
            WorldPosition if estimation possible, None otherwise
        """
        # Need at least one camera with detection
        valid_cameras = [
            (device_id, cameras[device_id], detections[device_id])
            for device_id in detections
            if device_id in cameras and detections[device_id].bbox
        ]

        if len(valid_cameras) == 0:
            return None

        if len(valid_cameras) == 1:
            # Single camera: estimate using ground plane intersection
            return self._single_camera_estimate(
                valid_cameras[0][1],
                valid_cameras[0][2],
                timestamp,
            )

        # Multiple cameras: triangulate
        return self._multi_camera_triangulate(valid_cameras, timestamp)

    def _single_camera_estimate(
        self,
        camera: CameraRegistration,
        detection: Detection,
        timestamp: float,
    ) -> Optional[WorldPosition]:
        """
        Estimate position from single camera using ground plane.

        Assumes predator is on the ground and uses vertical position
        in frame to estimate distance.
        """
        if not detection.bbox:
            return None

        # Get detection center
        cx = detection.bbox.center_x  # 0-1, left to right
        cy = detection.bbox.center_y  # 0-1, top to bottom

        # Convert to angle from camera center
        # Assuming camera is level (no tilt)
        h_angle = (cx - 0.5) * camera.fov_horizontal  # degrees from center
        v_angle = (cy - 0.5) * camera.fov_vertical  # degrees from center

        # Convert to radians
        h_rad = math.radians(h_angle)
        v_rad = math.radians(v_angle)
        heading_rad = math.radians(camera.heading)

        # Estimate distance using ground plane intersection
        # If object is lower in frame (higher cy), it's closer
        camera_height = camera.position_z - self.ground_plane_z

        if v_rad >= 0:  # Looking down or level
            # Distance based on vertical angle and height
            # tan(angle) = height / distance
            if v_rad > 0.01:  # Avoid division by very small numbers
                distance = camera_height / math.tan(v_rad)
            else:
                distance = 20.0  # Far away, use max estimate
        else:
            # Looking up, predator might be elevated (bird?)
            distance = 10.0  # Default estimate

        # Clamp distance to reasonable range
        distance = max(1.0, min(distance, 30.0))

        # Calculate world position
        world_x = camera.position_x + distance * math.sin(heading_rad + h_rad)
        world_y = camera.position_y + distance * math.cos(heading_rad + h_rad)

        # Lower confidence for single camera
        confidence = 0.5 * detection.confidence

        return WorldPosition(
            x=world_x,
            y=world_y,
            confidence=confidence,
            timestamp=timestamp,
        )

    def _multi_camera_triangulate(
        self,
        camera_data: list[tuple[str, CameraRegistration, Detection]],
        timestamp: float,
    ) -> Optional[WorldPosition]:
        """
        Triangulate position from multiple camera views.

        Uses least squares intersection of rays.
        """
        rays = []

        for device_id, camera, detection in camera_data:
            if not detection.bbox:
                continue

            # Calculate ray direction from camera through detection
            cx = detection.bbox.center_x
            cy = detection.bbox.center_y

            h_angle = (cx - 0.5) * camera.fov_horizontal
            v_angle = (cy - 0.5) * camera.fov_vertical

            h_rad = math.radians(h_angle)
            v_rad = math.radians(v_angle)
            heading_rad = math.radians(camera.heading)

            # Ray direction (normalized on ground plane)
            dx = math.sin(heading_rad + h_rad)
            dy = math.cos(heading_rad + h_rad)

            rays.append({
                "origin_x": camera.position_x,
                "origin_y": camera.position_y,
                "dir_x": dx,
                "dir_y": dy,
                "confidence": detection.confidence,
            })

        if len(rays) < 2:
            return None

        # Find intersection using weighted least squares
        # Minimize sum of squared distances from point to all rays
        position = self._intersect_rays(rays)

        if position is None:
            return None

        # Calculate confidence based on ray convergence
        total_confidence = sum(r["confidence"] for r in rays)
        avg_confidence = total_confidence / len(rays)

        # Boost confidence for multiple cameras
        confidence = min(1.0, avg_confidence * (1.0 + 0.2 * (len(rays) - 1)))

        return WorldPosition(
            x=position[0],
            y=position[1],
            confidence=confidence,
            timestamp=timestamp,
        )

    def _intersect_rays(
        self,
        rays: list[dict],
    ) -> Optional[tuple[float, float]]:
        """
        Find the point that minimizes distance to all rays.

        Uses weighted least squares approach.
        """
        if len(rays) < 2:
            return None

        # Build system of equations: Ax = b
        # For each ray: (I - d*d^T) * p = (I - d*d^T) * o
        # Where d is direction, o is origin, p is intersection point

        A = [[0.0, 0.0], [0.0, 0.0]]
        b = [0.0, 0.0]

        for ray in rays:
            dx, dy = ray["dir_x"], ray["dir_y"]
            ox, oy = ray["origin_x"], ray["origin_y"]
            w = ray["confidence"]  # Weight by confidence

            # (I - d*d^T) for 2D
            m00 = 1 - dx * dx
            m01 = -dx * dy
            m10 = -dx * dy
            m11 = 1 - dy * dy

            A[0][0] += w * m00
            A[0][1] += w * m01
            A[1][0] += w * m10
            A[1][1] += w * m11

            b[0] += w * (m00 * ox + m01 * oy)
            b[1] += w * (m10 * ox + m11 * oy)

        # Solve 2x2 system
        det = A[0][0] * A[1][1] - A[0][1] * A[1][0]

        if abs(det) < 1e-10:
            # Rays are parallel, no intersection
            return None

        x = (A[1][1] * b[0] - A[0][1] * b[1]) / det
        y = (A[0][0] * b[1] - A[1][0] * b[0]) / det

        return (x, y)

    def calculate_distance(
        self,
        pos1: WorldPosition,
        pos2: WorldPosition,
    ) -> float:
        """Calculate distance between two world positions."""
        dx = pos2.x - pos1.x
        dy = pos2.y - pos1.y
        return math.sqrt(dx * dx + dy * dy)

    def calculate_velocity(
        self,
        pos1: WorldPosition,
        pos2: WorldPosition,
    ) -> tuple[float, float]:
        """Calculate velocity vector between two positions."""
        dt = pos2.timestamp - pos1.timestamp
        if dt <= 0:
            return (0.0, 0.0)

        vx = (pos2.x - pos1.x) / dt
        vy = (pos2.y - pos1.y) / dt
        return (vx, vy)
