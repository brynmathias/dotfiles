"""
Simple motion detection using frame differencing.
Used to trigger frame uploads and optional edge inference.
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MotionConfig:
    enabled: bool = True
    sensitivity: int = 25  # 1-100 (lower = more sensitive)
    min_area: int = 500    # Minimum contour area to trigger
    cooldown_s: float = 2  # Seconds between motion events


class MotionDetector:
    """
    Detects motion in video frames using background subtraction.
    """

    def __init__(self, config: MotionConfig):
        self.config = config

        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=config.sensitivity,
            detectShadows=True,
        )

        self._last_motion_time: float = 0
        self._motion_callbacks: list[Callable[[np.ndarray, list], None]] = []
        self._frame_count = 0

    def process_frame(self, frame: np.ndarray, timestamp: float) -> tuple[bool, list]:
        """
        Process a frame for motion detection.

        Returns:
            tuple: (motion_detected, contours)
        """
        if not self.config.enabled:
            return False, []

        self._frame_count += 1

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Apply background subtraction
        fg_mask = self._bg_subtractor.apply(gray)

        # Remove shadows (marked as 127 in MOG2)
        fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]

        # Morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours by area
        significant_contours = [
            c for c in contours if cv2.contourArea(c) >= self.config.min_area
        ]

        # Skip first few frames while background model stabilizes
        if self._frame_count < 30:
            return False, []

        motion_detected = len(significant_contours) > 0

        # Check cooldown
        if motion_detected:
            time_since_last = timestamp - self._last_motion_time
            if time_since_last < self.config.cooldown_s:
                return False, significant_contours

            self._last_motion_time = timestamp
            logger.debug(f"Motion detected: {len(significant_contours)} regions")

            # Notify callbacks
            for callback in self._motion_callbacks:
                try:
                    callback(frame, significant_contours)
                except Exception as e:
                    logger.error(f"Motion callback error: {e}")

        return motion_detected, significant_contours

    def add_motion_callback(self, callback: Callable[[np.ndarray, list], None]):
        """Add a callback for motion events."""
        self._motion_callbacks.append(callback)

    def get_motion_regions(self, contours: list) -> list[tuple[int, int, int, int]]:
        """
        Get bounding boxes for motion regions.

        Returns:
            List of (x, y, w, h) tuples
        """
        return [cv2.boundingRect(c) for c in contours]

    def draw_motion_regions(
        self, frame: np.ndarray, contours: list, color: tuple = (0, 255, 0)
    ) -> np.ndarray:
        """
        Draw motion regions on a frame.

        Returns:
            Frame with motion regions drawn
        """
        result = frame.copy()

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

        return result

    def reset(self):
        """Reset the background model."""
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=self.config.sensitivity,
            detectShadows=True,
        )
        self._frame_count = 0
        logger.info("Motion detector reset")
