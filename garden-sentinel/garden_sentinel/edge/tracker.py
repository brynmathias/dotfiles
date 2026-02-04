"""
Object tracking module for edge devices.
Supports lightweight tracking algorithms that can run on TPU/CPU.
Used to track detected predators and control servo-mounted camera/water gun.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class TrackerType(Enum):
    CSRT = "csrt"           # More accurate, slower
    KCF = "kcf"             # Good balance
    MOSSE = "mosse"         # Fastest, less accurate
    MIL = "mil"             # Medium
    MEDIANFLOW = "medianflow"  # Good for predictable motion


@dataclass
class TrackingConfig:
    tracker_type: TrackerType = TrackerType.KCF
    max_frames_without_detection: int = 30  # Re-detect after this many frames
    min_confidence: float = 0.3
    smoothing_factor: float = 0.7  # For position smoothing


@dataclass
class TrackingState:
    is_tracking: bool = False
    target_class: Optional[str] = None
    bbox: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h in pixels
    center: Optional[Tuple[float, float]] = None  # Normalized 0-1
    confidence: float = 0.0
    frames_tracked: int = 0
    frames_since_detection: int = 0


class ObjectTracker:
    """
    Lightweight object tracker for edge devices.
    Tracks a single target and provides position updates for servo control.
    """

    def __init__(self, config: TrackingConfig):
        self.config = config
        self._tracker = None
        self._state = TrackingState()
        self._frame_size: Optional[Tuple[int, int]] = None
        self._smoothed_center: Optional[Tuple[float, float]] = None

    def _create_tracker(self):
        """Create OpenCV tracker based on config."""
        tracker_map = {
            TrackerType.CSRT: cv2.TrackerCSRT_create,
            TrackerType.KCF: cv2.TrackerKCF_create,
            TrackerType.MIL: cv2.TrackerMIL_create,
        }

        # MOSSE and MEDIANFLOW might not be available in all OpenCV versions
        if hasattr(cv2, 'legacy'):
            tracker_map[TrackerType.MOSSE] = cv2.legacy.TrackerMOSSE_create
            tracker_map[TrackerType.MEDIANFLOW] = cv2.legacy.TrackerMedianFlow_create

        creator = tracker_map.get(self.config.tracker_type)
        if creator:
            return creator()
        else:
            logger.warning(f"Tracker {self.config.tracker_type} not available, using KCF")
            return cv2.TrackerKCF_create()

    def start_tracking(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],  # Normalized x, y, w, h
        target_class: str,
    ) -> bool:
        """
        Start tracking a target.

        Args:
            frame: BGR image
            bbox: Normalized bounding box (x, y, w, h) in range 0-1
            target_class: Class name of the target

        Returns:
            True if tracking started successfully
        """
        height, width = frame.shape[:2]
        self._frame_size = (width, height)

        # Convert normalized bbox to pixels
        x = int(bbox[0] * width)
        y = int(bbox[1] * height)
        w = int(bbox[2] * width)
        h = int(bbox[3] * height)

        # Ensure bbox is within frame
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))

        pixel_bbox = (x, y, w, h)

        # Create and initialize tracker
        self._tracker = self._create_tracker()

        try:
            success = self._tracker.init(frame, pixel_bbox)
        except Exception as e:
            logger.error(f"Failed to initialize tracker: {e}")
            return False

        if success:
            self._state = TrackingState(
                is_tracking=True,
                target_class=target_class,
                bbox=pixel_bbox,
                center=((x + w / 2) / width, (y + h / 2) / height),
                confidence=1.0,
                frames_tracked=1,
                frames_since_detection=0,
            )
            self._smoothed_center = self._state.center
            logger.info(f"Started tracking {target_class} at {pixel_bbox}")
            return True

        return False

    def update(self, frame: np.ndarray) -> TrackingState:
        """
        Update tracker with new frame.

        Args:
            frame: BGR image

        Returns:
            Current tracking state
        """
        if not self._state.is_tracking or self._tracker is None:
            return self._state

        height, width = frame.shape[:2]

        try:
            success, bbox = self._tracker.update(frame)
        except Exception as e:
            logger.error(f"Tracker update failed: {e}")
            success = False
            bbox = None

        if success and bbox is not None:
            x, y, w, h = [int(v) for v in bbox]

            # Update state
            self._state.bbox = (x, y, w, h)
            self._state.frames_tracked += 1
            self._state.frames_since_detection += 1

            # Calculate center (normalized)
            raw_center = ((x + w / 2) / width, (y + h / 2) / height)

            # Apply smoothing
            if self._smoothed_center:
                alpha = self.config.smoothing_factor
                self._smoothed_center = (
                    alpha * self._smoothed_center[0] + (1 - alpha) * raw_center[0],
                    alpha * self._smoothed_center[1] + (1 - alpha) * raw_center[1],
                )
            else:
                self._smoothed_center = raw_center

            self._state.center = self._smoothed_center

            # Decay confidence over time without re-detection
            decay_rate = 0.02
            self._state.confidence = max(
                self.config.min_confidence,
                1.0 - (self._state.frames_since_detection * decay_rate)
            )

            # Check if we should request re-detection
            if self._state.frames_since_detection >= self.config.max_frames_without_detection:
                logger.debug("Tracking confidence low, requesting re-detection")

        else:
            # Tracking failed
            self._state.is_tracking = False
            self._state.confidence = 0.0
            logger.info("Tracking lost")

        return self._state

    def reinforce(
        self,
        bbox: Tuple[float, float, float, float],  # Normalized
        confidence: float,
    ):
        """
        Reinforce tracking with a new detection.
        Call this when the detector confirms the tracked object.
        """
        if not self._state.is_tracking or self._frame_size is None:
            return

        width, height = self._frame_size

        # Update with new detection
        x = int(bbox[0] * width)
        y = int(bbox[1] * height)
        w = int(bbox[2] * width)
        h = int(bbox[3] * height)

        self._state.bbox = (x, y, w, h)
        self._state.center = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
        self._state.confidence = confidence
        self._state.frames_since_detection = 0
        self._smoothed_center = self._state.center

        logger.debug(f"Tracking reinforced with confidence {confidence:.2f}")

    def stop_tracking(self):
        """Stop tracking."""
        self._state = TrackingState()
        self._tracker = None
        self._smoothed_center = None
        logger.info("Tracking stopped")

    def get_target_offset(self) -> Optional[Tuple[float, float]]:
        """
        Get the offset of the target from frame center.

        Returns:
            (x_offset, y_offset) where:
            - Positive x = target is right of center
            - Positive y = target is below center
            - Values range from -0.5 to 0.5
            Returns None if not tracking.
        """
        if not self._state.is_tracking or self._state.center is None:
            return None

        # Center is at (0.5, 0.5), so offset is center - 0.5
        x_offset = self._state.center[0] - 0.5
        y_offset = self._state.center[1] - 0.5

        return (x_offset, y_offset)

    def draw_tracking(self, frame: np.ndarray) -> np.ndarray:
        """Draw tracking visualization on frame."""
        if not self._state.is_tracking or self._state.bbox is None:
            return frame

        result = frame.copy()
        x, y, w, h = self._state.bbox

        # Color based on confidence
        if self._state.confidence > 0.7:
            color = (0, 255, 0)  # Green
        elif self._state.confidence > 0.4:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 165, 255)  # Orange

        # Draw bounding box
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

        # Draw center crosshair
        cx, cy = x + w // 2, y + h // 2
        cv2.drawMarker(result, (cx, cy), color, cv2.MARKER_CROSS, 20, 2)

        # Draw label
        label = f"Tracking: {self._state.target_class} ({self._state.confidence:.0%})"
        cv2.putText(result, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw frame center reference
        height, width = frame.shape[:2]
        cv2.drawMarker(
            result, (width // 2, height // 2),
            (128, 128, 128), cv2.MARKER_CROSS, 30, 1
        )

        return result

    @property
    def state(self) -> TrackingState:
        return self._state

    @property
    def is_tracking(self) -> bool:
        return self._state.is_tracking

    @property
    def needs_redetection(self) -> bool:
        """Check if tracking needs reinforcement from detector."""
        return (
            self._state.is_tracking and
            self._state.frames_since_detection >= self.config.max_frames_without_detection
        )
