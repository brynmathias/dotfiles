"""
Camera module for Raspberry Pi 5 using picamera2.
Handles video capture and frame management.
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
class CameraConfig:
    width: int = 1920
    height: int = 1080
    fps: int = 30
    rotation: int = 0
    hflip: bool = False
    vflip: bool = False
    buffer_count: int = 4
    auto_exposure: bool = True
    exposure_time_us: Optional[int] = None
    iso: Optional[int] = None


class Camera:
    """
    Camera capture handler using picamera2 for Raspberry Pi 5.
    Falls back to OpenCV for testing on non-Pi systems.
    """

    def __init__(self, config: CameraConfig):
        self.config = config
        self._picamera = None
        self._cv_camera = None
        self._frame = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._capture_thread = None
        self._frame_callbacks: list[Callable[[np.ndarray, float], None]] = []
        self._frame_count = 0
        self._start_time = 0

    def start(self):
        """Start the camera and capture thread."""
        if self._running:
            return

        self._running = True
        self._start_time = time.time()

        # Try picamera2 first (Raspberry Pi)
        if self._init_picamera():
            logger.info("Initialized picamera2 for Raspberry Pi")
            self._capture_thread = threading.Thread(
                target=self._capture_loop_picamera, daemon=True
            )
        else:
            # Fall back to OpenCV (for testing on other systems)
            if self._init_opencv():
                logger.info("Initialized OpenCV camera (fallback mode)")
                self._capture_thread = threading.Thread(
                    target=self._capture_loop_opencv, daemon=True
                )
            else:
                logger.error("No camera available")
                self._running = False
                return

        self._capture_thread.start()
        logger.info(
            f"Camera started: {self.config.width}x{self.config.height}@{self.config.fps}fps"
        )

    def stop(self):
        """Stop the camera and cleanup."""
        self._running = False

        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)

        if self._picamera:
            try:
                self._picamera.stop()
                self._picamera.close()
            except Exception as e:
                logger.warning(f"Error closing picamera: {e}")
            self._picamera = None

        if self._cv_camera:
            self._cv_camera.release()
            self._cv_camera = None

        logger.info("Camera stopped")

    def _init_picamera(self) -> bool:
        """Initialize picamera2 for Raspberry Pi."""
        try:
            from picamera2 import Picamera2

            self._picamera = Picamera2()

            # Configure camera
            config = self._picamera.create_video_configuration(
                main={"size": (self.config.width, self.config.height), "format": "RGB888"},
                buffer_count=self.config.buffer_count,
            )

            # Apply rotation and flip
            transform = None
            if self.config.hflip or self.config.vflip or self.config.rotation:
                from libcamera import Transform

                transform = Transform(
                    hflip=self.config.hflip,
                    vflip=self.config.vflip,
                )
                # Note: rotation is handled separately in picamera2

            if transform:
                config["transform"] = transform

            self._picamera.configure(config)

            # Set controls
            controls = {}
            if not self.config.auto_exposure and self.config.exposure_time_us:
                controls["ExposureTime"] = self.config.exposure_time_us
            if self.config.iso:
                controls["AnalogueGain"] = self.config.iso / 100

            if controls:
                self._picamera.set_controls(controls)

            self._picamera.start()
            return True

        except ImportError:
            logger.debug("picamera2 not available")
            return False
        except Exception as e:
            logger.debug(f"Failed to initialize picamera2: {e}")
            return False

    def _init_opencv(self) -> bool:
        """Initialize OpenCV camera as fallback."""
        try:
            self._cv_camera = cv2.VideoCapture(0)
            if not self._cv_camera.isOpened():
                return False

            self._cv_camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self._cv_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self._cv_camera.set(cv2.CAP_PROP_FPS, self.config.fps)

            return True
        except Exception as e:
            logger.debug(f"Failed to initialize OpenCV camera: {e}")
            return False

    def _capture_loop_picamera(self):
        """Capture loop for picamera2."""
        frame_interval = 1.0 / self.config.fps

        while self._running:
            try:
                frame = self._picamera.capture_array()
                timestamp = time.time()

                # Convert RGB to BGR for OpenCV compatibility
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Apply rotation if needed
                if self.config.rotation == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif self.config.rotation == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif self.config.rotation == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                self._update_frame(frame, timestamp)

                # Rate limiting
                elapsed = time.time() - timestamp
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)

            except Exception as e:
                logger.error(f"Capture error: {e}")
                time.sleep(0.1)

    def _capture_loop_opencv(self):
        """Capture loop for OpenCV camera."""
        frame_interval = 1.0 / self.config.fps

        while self._running:
            try:
                ret, frame = self._cv_camera.read()
                timestamp = time.time()

                if not ret:
                    logger.warning("Failed to capture frame")
                    time.sleep(0.1)
                    continue

                # Apply transformations
                if self.config.hflip:
                    frame = cv2.flip(frame, 1)
                if self.config.vflip:
                    frame = cv2.flip(frame, 0)
                if self.config.rotation == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif self.config.rotation == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif self.config.rotation == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                self._update_frame(frame, timestamp)

                # Rate limiting
                elapsed = time.time() - timestamp
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)

            except Exception as e:
                logger.error(f"Capture error: {e}")
                time.sleep(0.1)

    def _update_frame(self, frame: np.ndarray, timestamp: float):
        """Update the current frame and notify callbacks."""
        with self._frame_lock:
            self._frame = frame.copy()
            self._frame_count += 1

        # Notify callbacks
        for callback in self._frame_callbacks:
            try:
                callback(frame, timestamp)
            except Exception as e:
                logger.error(f"Frame callback error: {e}")

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the current frame (thread-safe copy)."""
        with self._frame_lock:
            if self._frame is not None:
                return self._frame.copy()
        return None

    def get_jpeg(self, quality: int = 85) -> Optional[bytes]:
        """Get the current frame as JPEG bytes."""
        frame = self.get_frame()
        if frame is not None:
            ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ret:
                return jpeg.tobytes()
        return None

    def add_frame_callback(self, callback: Callable[[np.ndarray, float], None]):
        """Add a callback to be called on each new frame."""
        self._frame_callbacks.append(callback)

    def remove_frame_callback(self, callback: Callable[[np.ndarray, float], None]):
        """Remove a frame callback."""
        if callback in self._frame_callbacks:
            self._frame_callbacks.remove(callback)

    @property
    def fps(self) -> float:
        """Get the actual FPS."""
        if self._start_time == 0:
            return 0
        elapsed = time.time() - self._start_time
        if elapsed > 0:
            return self._frame_count / elapsed
        return 0

    @property
    def is_running(self) -> bool:
        """Check if the camera is running."""
        return self._running
