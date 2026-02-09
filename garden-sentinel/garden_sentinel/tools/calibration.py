"""
Camera calibration tool for Garden Sentinel.

Provides tools for:
- Intrinsic camera calibration (lens distortion)
- Extrinsic calibration (camera position/orientation in world)
- Ground plane estimation
- Pixel-to-world coordinate transformation
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import json
import math
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CalibrationPoint:
    """A point with known image and world coordinates."""
    image_x: float  # Pixel coordinates
    image_y: float
    world_x: float  # Real-world coordinates (meters)
    world_y: float
    world_z: float = 0.0  # Assume ground plane by default


@dataclass
class CameraIntrinsics:
    """Intrinsic camera parameters."""
    fx: float  # Focal length x (pixels)
    fy: float  # Focal length y (pixels)
    cx: float  # Principal point x
    cy: float  # Principal point y
    k1: float = 0.0  # Radial distortion coefficient 1
    k2: float = 0.0  # Radial distortion coefficient 2
    p1: float = 0.0  # Tangential distortion coefficient 1
    p2: float = 0.0  # Tangential distortion coefficient 2
    width: int = 1920
    height: int = 1080

    def to_matrix(self) -> List[List[float]]:
        """Return camera matrix as 3x3 list."""
        return [
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1],
        ]

    def distortion_coeffs(self) -> List[float]:
        """Return distortion coefficients."""
        return [self.k1, self.k2, self.p1, self.p2]


@dataclass
class CameraExtrinsics:
    """Extrinsic camera parameters (pose in world)."""
    x: float  # Position in world (meters)
    y: float
    z: float  # Height above ground
    pan: float  # Rotation around vertical axis (degrees)
    tilt: float  # Rotation around horizontal axis (degrees)
    roll: float = 0.0  # Rotation around optical axis (degrees)

    def to_rotation_matrix(self) -> List[List[float]]:
        """Convert Euler angles to rotation matrix."""
        # Convert to radians
        pan_rad = math.radians(self.pan)
        tilt_rad = math.radians(self.tilt)
        roll_rad = math.radians(self.roll)

        # Rotation matrices
        # Pan (yaw) - rotation around Z
        c_pan, s_pan = math.cos(pan_rad), math.sin(pan_rad)
        R_pan = [
            [c_pan, -s_pan, 0],
            [s_pan, c_pan, 0],
            [0, 0, 1],
        ]

        # Tilt (pitch) - rotation around X
        c_tilt, s_tilt = math.cos(tilt_rad), math.sin(tilt_rad)
        R_tilt = [
            [1, 0, 0],
            [0, c_tilt, -s_tilt],
            [0, s_tilt, c_tilt],
        ]

        # Roll - rotation around Y
        c_roll, s_roll = math.cos(roll_rad), math.sin(roll_rad)
        R_roll = [
            [c_roll, 0, s_roll],
            [0, 1, 0],
            [-s_roll, 0, c_roll],
        ]

        # Combined rotation: R = R_pan * R_tilt * R_roll
        def matmul(A, B):
            return [
                [sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)]
                for i in range(3)
            ]

        R = matmul(matmul(R_pan, R_tilt), R_roll)
        return R

    def translation_vector(self) -> List[float]:
        """Return translation vector."""
        return [self.x, self.y, self.z]


@dataclass
class GroundPlane:
    """Ground plane definition in world coordinates."""
    # Plane equation: ax + by + cz + d = 0
    a: float = 0.0
    b: float = 0.0
    c: float = 1.0  # Default: horizontal plane (z=0)
    d: float = 0.0

    @classmethod
    def horizontal(cls, height: float = 0.0) -> "GroundPlane":
        """Create a horizontal ground plane at given height."""
        return cls(a=0, b=0, c=1, d=-height)

    def point_height(self, x: float, y: float) -> float:
        """Calculate z coordinate on plane for given x, y."""
        if abs(self.c) < 1e-6:
            return 0.0
        return -(self.a * x + self.b * y + self.d) / self.c


@dataclass
class PerspectiveTransform:
    """2D perspective transformation (homography)."""
    matrix: List[List[float]] = field(default_factory=lambda: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])

    def transform_point(self, x: float, y: float) -> Tuple[float, float]:
        """Transform a point using the homography."""
        H = self.matrix
        w = H[2][0] * x + H[2][1] * y + H[2][2]
        if abs(w) < 1e-10:
            return (0.0, 0.0)

        tx = (H[0][0] * x + H[0][1] * y + H[0][2]) / w
        ty = (H[1][0] * x + H[1][1] * y + H[1][2]) / w
        return (tx, ty)

    def inverse(self) -> "PerspectiveTransform":
        """Return inverse transformation."""
        H = self.matrix
        # Calculate determinant
        det = (
            H[0][0] * (H[1][1] * H[2][2] - H[1][2] * H[2][1])
            - H[0][1] * (H[1][0] * H[2][2] - H[1][2] * H[2][0])
            + H[0][2] * (H[1][0] * H[2][1] - H[1][1] * H[2][0])
        )

        if abs(det) < 1e-10:
            return PerspectiveTransform()  # Identity

        # Calculate inverse using adjugate matrix
        inv = [
            [
                (H[1][1] * H[2][2] - H[1][2] * H[2][1]) / det,
                (H[0][2] * H[2][1] - H[0][1] * H[2][2]) / det,
                (H[0][1] * H[1][2] - H[0][2] * H[1][1]) / det,
            ],
            [
                (H[1][2] * H[2][0] - H[1][0] * H[2][2]) / det,
                (H[0][0] * H[2][2] - H[0][2] * H[2][0]) / det,
                (H[0][2] * H[1][0] - H[0][0] * H[1][2]) / det,
            ],
            [
                (H[1][0] * H[2][1] - H[1][1] * H[2][0]) / det,
                (H[0][1] * H[2][0] - H[0][0] * H[2][1]) / det,
                (H[0][0] * H[1][1] - H[0][1] * H[1][0]) / det,
            ],
        ]

        return PerspectiveTransform(matrix=inv)


@dataclass
class CalibrationResult:
    """Complete calibration result for a camera."""
    device_id: str
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    ground_plane: GroundPlane
    homography: PerspectiveTransform
    calibration_points: List[CalibrationPoint]
    reprojection_error: float = 0.0
    timestamp: float = 0.0

    def pixel_to_world(self, px: float, py: float) -> Tuple[float, float, float]:
        """Convert pixel coordinates to world coordinates on ground plane."""
        # Apply homography to get ground plane coordinates
        wx, wy = self.homography.transform_point(px, py)
        wz = self.ground_plane.point_height(wx, wy)
        return (wx, wy, wz)

    def world_to_pixel(self, wx: float, wy: float, wz: float = None) -> Tuple[float, float]:
        """Convert world coordinates to pixel coordinates."""
        if wz is None:
            wz = self.ground_plane.point_height(wx, wy)

        # Use inverse homography for ground plane
        inv_h = self.homography.inverse()
        px, py = inv_h.transform_point(wx, wy)
        return (px, py)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "device_id": self.device_id,
            "intrinsics": {
                "fx": self.intrinsics.fx,
                "fy": self.intrinsics.fy,
                "cx": self.intrinsics.cx,
                "cy": self.intrinsics.cy,
                "k1": self.intrinsics.k1,
                "k2": self.intrinsics.k2,
                "p1": self.intrinsics.p1,
                "p2": self.intrinsics.p2,
                "width": self.intrinsics.width,
                "height": self.intrinsics.height,
            },
            "extrinsics": {
                "x": self.extrinsics.x,
                "y": self.extrinsics.y,
                "z": self.extrinsics.z,
                "pan": self.extrinsics.pan,
                "tilt": self.extrinsics.tilt,
                "roll": self.extrinsics.roll,
            },
            "ground_plane": {
                "a": self.ground_plane.a,
                "b": self.ground_plane.b,
                "c": self.ground_plane.c,
                "d": self.ground_plane.d,
            },
            "homography": self.homography.matrix,
            "calibration_points": [
                {
                    "image_x": p.image_x,
                    "image_y": p.image_y,
                    "world_x": p.world_x,
                    "world_y": p.world_y,
                    "world_z": p.world_z,
                }
                for p in self.calibration_points
            ],
            "reprojection_error": self.reprojection_error,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationResult":
        """Deserialize from dictionary."""
        return cls(
            device_id=data["device_id"],
            intrinsics=CameraIntrinsics(**data["intrinsics"]),
            extrinsics=CameraExtrinsics(**data["extrinsics"]),
            ground_plane=GroundPlane(**data["ground_plane"]),
            homography=PerspectiveTransform(matrix=data["homography"]),
            calibration_points=[
                CalibrationPoint(**p) for p in data["calibration_points"]
            ],
            reprojection_error=data.get("reprojection_error", 0.0),
            timestamp=data.get("timestamp", 0.0),
        )


class CameraCalibrator:
    """
    Interactive camera calibration tool.

    Supports:
    - Manual point correspondence calibration
    - Automatic checkerboard detection (when OpenCV available)
    - Homography estimation from 4+ points
    """

    MIN_POINTS = 4  # Minimum points for homography

    def __init__(self, device_id: str, width: int = 1920, height: int = 1080):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.calibration_points: List[CalibrationPoint] = []
        self._opencv_available = self._check_opencv()

    def _check_opencv(self) -> bool:
        """Check if OpenCV is available."""
        try:
            import cv2
            return True
        except ImportError:
            return False

    def add_point(
        self,
        image_x: float,
        image_y: float,
        world_x: float,
        world_y: float,
        world_z: float = 0.0,
    ):
        """Add a calibration point correspondence."""
        point = CalibrationPoint(
            image_x=image_x,
            image_y=image_y,
            world_x=world_x,
            world_y=world_y,
            world_z=world_z,
        )
        self.calibration_points.append(point)
        logger.info(f"Added calibration point: image({image_x}, {image_y}) -> world({world_x}, {world_y}, {world_z})")

    def clear_points(self):
        """Clear all calibration points."""
        self.calibration_points.clear()

    def estimate_homography(self) -> PerspectiveTransform:
        """
        Estimate homography from calibration points.

        Uses Direct Linear Transform (DLT) algorithm.
        Requires at least 4 points on the same plane.
        """
        if len(self.calibration_points) < self.MIN_POINTS:
            raise ValueError(f"Need at least {self.MIN_POINTS} points, have {len(self.calibration_points)}")

        # Build the DLT matrix
        # For each point correspondence (x, y) -> (X, Y):
        # [X Y 1 0 0 0 -xX -xY -x]   [h1]
        # [0 0 0 X Y 1 -yX -yY -y] * [h2] = 0
        #                            [...]
        #                            [h9]

        n = len(self.calibration_points)
        A = []

        for p in self.calibration_points:
            X, Y = p.world_x, p.world_y
            x, y = p.image_x, p.image_y

            A.append([X, Y, 1, 0, 0, 0, -x * X, -x * Y, -x])
            A.append([0, 0, 0, X, Y, 1, -y * X, -y * Y, -y])

        # Solve using SVD (simplified without numpy)
        # For a proper implementation, use numpy or OpenCV
        # Here we use a simplified approach for 4 points

        if self._opencv_available:
            return self._estimate_homography_opencv()

        # Fallback: use simplified 4-point solution
        return self._estimate_homography_simple()

    def _estimate_homography_opencv(self) -> PerspectiveTransform:
        """Estimate homography using OpenCV."""
        import cv2
        import numpy as np

        src_points = np.array(
            [[p.world_x, p.world_y] for p in self.calibration_points],
            dtype=np.float32
        )
        dst_points = np.array(
            [[p.image_x, p.image_y] for p in self.calibration_points],
            dtype=np.float32
        )

        H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

        if H is None:
            raise ValueError("Failed to estimate homography")

        # Invert to get world->image (we store image->world)
        H_inv = np.linalg.inv(H)
        return PerspectiveTransform(matrix=H_inv.tolist())

    def _estimate_homography_simple(self) -> PerspectiveTransform:
        """
        Simple 4-point homography estimation.

        This is a simplified version without full SVD.
        For production, use OpenCV or numpy.
        """
        if len(self.calibration_points) < 4:
            raise ValueError("Need at least 4 points")

        # Use first 4 points
        pts = self.calibration_points[:4]

        # Simplified bilinear interpolation approach
        # This is an approximation for small perspective distortion

        # Calculate scale factors
        img_width = max(p.image_x for p in pts) - min(p.image_x for p in pts)
        img_height = max(p.image_y for p in pts) - min(p.image_y for p in pts)
        world_width = max(p.world_x for p in pts) - min(p.world_x for p in pts)
        world_height = max(p.world_y for p in pts) - min(p.world_y for p in pts)

        scale_x = world_width / img_width if img_width > 0 else 1
        scale_y = world_height / img_height if img_height > 0 else 1

        # Calculate offset
        offset_x = min(p.world_x for p in pts) - min(p.image_x for p in pts) * scale_x
        offset_y = min(p.world_y for p in pts) - min(p.image_y for p in pts) * scale_y

        # Create affine-like homography (simplified)
        H = [
            [scale_x, 0, offset_x],
            [0, scale_y, offset_y],
            [0, 0, 1],
        ]

        return PerspectiveTransform(matrix=H)

    def calibrate(
        self,
        camera_position: Optional[Tuple[float, float, float]] = None,
        camera_rotation: Optional[Tuple[float, float, float]] = None,
    ) -> CalibrationResult:
        """
        Perform full calibration.

        Args:
            camera_position: (x, y, z) position in world coordinates
            camera_rotation: (pan, tilt, roll) in degrees

        Returns:
            CalibrationResult with all calibration data
        """
        import time

        # Estimate homography
        homography = self.estimate_homography()

        # Calculate reprojection error
        error = self._calculate_reprojection_error(homography)

        # Create intrinsics (default or from prior calibration)
        intrinsics = CameraIntrinsics(
            fx=self.width,  # Approximate focal length
            fy=self.width,
            cx=self.width / 2,
            cy=self.height / 2,
            width=self.width,
            height=self.height,
        )

        # Create extrinsics
        if camera_position is None:
            camera_position = (0, 0, 2.5)  # Default 2.5m height
        if camera_rotation is None:
            camera_rotation = (0, -30, 0)  # Default 30 degree tilt down

        extrinsics = CameraExtrinsics(
            x=camera_position[0],
            y=camera_position[1],
            z=camera_position[2],
            pan=camera_rotation[0],
            tilt=camera_rotation[1],
            roll=camera_rotation[2],
        )

        result = CalibrationResult(
            device_id=self.device_id,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            ground_plane=GroundPlane.horizontal(),
            homography=homography,
            calibration_points=list(self.calibration_points),
            reprojection_error=error,
            timestamp=time.time(),
        )

        logger.info(f"Calibration complete for {self.device_id}. Reprojection error: {error:.2f} pixels")
        return result

    def _calculate_reprojection_error(self, homography: PerspectiveTransform) -> float:
        """Calculate average reprojection error in pixels."""
        if not self.calibration_points:
            return 0.0

        total_error = 0.0
        inv_h = homography.inverse()

        for p in self.calibration_points:
            # Project world point to image
            proj_x, proj_y = inv_h.transform_point(p.world_x, p.world_y)

            # Calculate error
            error = math.sqrt((proj_x - p.image_x) ** 2 + (proj_y - p.image_y) ** 2)
            total_error += error

        return total_error / len(self.calibration_points)

    def save_calibration(self, result: CalibrationResult, path: Path):
        """Save calibration result to file."""
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Saved calibration to {path}")

    @staticmethod
    def load_calibration(path: Path) -> CalibrationResult:
        """Load calibration result from file."""
        with open(path, "r") as f:
            data = json.load(f)
        return CalibrationResult.from_dict(data)


class InteractiveCalibrator:
    """
    Interactive calibration UI helper.

    Provides methods for building a CLI or GUI calibration interface.
    """

    def __init__(self, calibrator: CameraCalibrator):
        self.calibrator = calibrator
        self.current_image_point: Optional[Tuple[float, float]] = None

    def click_image(self, x: float, y: float):
        """Record an image point click."""
        self.current_image_point = (x, y)
        return f"Image point recorded: ({x}, {y})"

    def set_world_coords(self, x: float, y: float, z: float = 0.0) -> str:
        """Set world coordinates for the current image point."""
        if self.current_image_point is None:
            return "Error: No image point selected"

        self.calibrator.add_point(
            image_x=self.current_image_point[0],
            image_y=self.current_image_point[1],
            world_x=x,
            world_y=y,
            world_z=z,
        )
        self.current_image_point = None
        return f"Point added: image -> world({x}, {y}, {z})"

    def get_status(self) -> Dict[str, Any]:
        """Get current calibration status."""
        return {
            "device_id": self.calibrator.device_id,
            "num_points": len(self.calibrator.calibration_points),
            "can_calibrate": len(self.calibrator.calibration_points) >= CameraCalibrator.MIN_POINTS,
            "current_image_point": self.current_image_point,
            "points": [
                {
                    "image": (p.image_x, p.image_y),
                    "world": (p.world_x, p.world_y, p.world_z),
                }
                for p in self.calibrator.calibration_points
            ],
        }

    def run_calibration(
        self,
        camera_position: Optional[Tuple[float, float, float]] = None,
        camera_rotation: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[bool, str, Optional[CalibrationResult]]:
        """
        Run calibration and return result.

        Returns:
            Tuple of (success, message, result)
        """
        if len(self.calibrator.calibration_points) < CameraCalibrator.MIN_POINTS:
            return (
                False,
                f"Need at least {CameraCalibrator.MIN_POINTS} points",
                None,
            )

        try:
            result = self.calibrator.calibrate(camera_position, camera_rotation)
            return (
                True,
                f"Calibration successful. Error: {result.reprojection_error:.2f}px",
                result,
            )
        except Exception as e:
            return (False, f"Calibration failed: {e}", None)


def create_calibration_wizard() -> str:
    """
    Return instructions for the calibration wizard.
    """
    return """
Camera Calibration Wizard
=========================

This tool helps you calibrate your camera for accurate position tracking.

Steps:
1. Place markers at known positions in your garden (corners of zones, etc.)
2. Measure the real-world coordinates of each marker
3. Click on each marker in the camera image
4. Enter the corresponding world coordinates
5. Repeat for at least 4 points (more is better)
6. Run calibration

Tips:
- Use points spread across the entire field of view
- Avoid collinear points (don't put all points in a line)
- Place points at different distances from the camera
- Accuracy improves with more points
- Re-calibrate if the camera is moved

Commands:
- add <img_x> <img_y> <world_x> <world_y> - Add a calibration point
- list - Show current points
- clear - Remove all points
- calibrate - Run calibration
- save <path> - Save calibration to file
- load <path> - Load calibration from file
- test <img_x> <img_y> - Test pixel to world conversion
"""
