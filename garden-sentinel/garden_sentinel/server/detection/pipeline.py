"""
Detection pipeline for processing video frames from multiple edge devices.
Handles queuing, batching, and result dispatching.
"""

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional
import uuid

import cv2
import numpy as np

from .detector import PredatorDetector, DetectionConfig

logger = logging.getLogger(__name__)

from garden_sentinel.shared import Detection, ThreatLevel, AlertEvent


@dataclass
class FrameInput:
    """Frame input for the detection pipeline."""
    device_id: str
    frame: np.ndarray
    timestamp: float
    frame_id: str = ""

    def __post_init__(self):
        if not self.frame_id:
            self.frame_id = str(uuid.uuid4())[:8]


@dataclass
class DetectionResult:
    """Result from the detection pipeline."""
    device_id: str
    frame_id: str
    timestamp: float
    detections: list[Detection]
    inference_time_ms: float
    max_threat_level: Optional[ThreatLevel]
    annotated_frame: Optional[np.ndarray] = None


class DetectionPipeline:
    """
    Asynchronous detection pipeline for processing frames from multiple cameras.
    """

    def __init__(
        self,
        config: DetectionConfig,
        max_queue_size: int = 100,
        num_workers: int = 1,
    ):
        self.config = config
        self.max_queue_size = max_queue_size
        self.num_workers = num_workers

        self._detector = PredatorDetector(config)
        self._frame_queue: queue.Queue[FrameInput] = queue.Queue(maxsize=max_queue_size)
        self._result_callbacks: list[Callable[[DetectionResult], None]] = []
        self._alert_callbacks: list[Callable[[AlertEvent], None]] = []

        self._workers: list[threading.Thread] = []
        self._running = False

        # Statistics
        self._stats = {
            "frames_processed": 0,
            "total_inference_time_ms": 0,
            "detections_count": 0,
            "alerts_triggered": 0,
            "queue_drops": 0,
        }
        self._stats_lock = threading.Lock()

        # Alert cooldown tracking per device
        self._last_alert_time: dict[str, float] = {}
        self._alert_cooldown_s = 30

    def start(self) -> bool:
        """Start the detection pipeline."""
        if not self._detector.initialize():
            logger.error("Failed to initialize detector")
            return False

        self._running = True

        # Start worker threads
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"detection-worker-{i}",
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

        logger.info(f"Detection pipeline started with {self.num_workers} workers")
        return True

    def stop(self):
        """Stop the detection pipeline."""
        self._running = False

        # Clear queue to unblock workers
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

        # Wait for workers to finish
        for worker in self._workers:
            worker.join(timeout=2.0)

        self._workers.clear()
        logger.info("Detection pipeline stopped")

    def submit_frame(self, device_id: str, frame: np.ndarray, timestamp: Optional[float] = None) -> bool:
        """
        Submit a frame for detection.

        Returns:
            True if frame was queued, False if queue is full
        """
        if not self._running:
            return False

        if timestamp is None:
            timestamp = time.time()

        frame_input = FrameInput(
            device_id=device_id,
            frame=frame,
            timestamp=timestamp,
        )

        try:
            self._frame_queue.put_nowait(frame_input)
            return True
        except queue.Full:
            with self._stats_lock:
                self._stats["queue_drops"] += 1
            logger.warning(f"Frame queue full, dropping frame from {device_id}")
            return False

    def submit_frame_bytes(self, device_id: str, jpeg_bytes: bytes, timestamp: Optional[float] = None) -> bool:
        """
        Submit a JPEG frame for detection.

        Returns:
            True if frame was queued, False if failed
        """
        try:
            # Decode JPEG
            nparr = np.frombuffer(jpeg_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                logger.warning(f"Failed to decode frame from {device_id}")
                return False

            return self.submit_frame(device_id, frame, timestamp)

        except Exception as e:
            logger.error(f"Error decoding frame: {e}")
            return False

    def _worker_loop(self):
        """Worker thread for processing frames."""
        while self._running:
            try:
                # Get frame from queue with timeout
                try:
                    frame_input = self._frame_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # Run detection
                detections, inference_time = self._detector.detect(frame_input.frame)

                # Determine max threat level
                max_threat = None
                if detections:
                    threat_levels = [d.threat_level for d in detections if d.threat_level]
                    if threat_levels:
                        # Sort by severity
                        severity_order = [ThreatLevel.LOW, ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL]
                        max_threat = max(threat_levels, key=lambda t: severity_order.index(t))

                # Create annotated frame if there are detections
                annotated = None
                if detections:
                    annotated = self._detector.draw_detections(frame_input.frame, detections)

                # Create result
                result = DetectionResult(
                    device_id=frame_input.device_id,
                    frame_id=frame_input.frame_id,
                    timestamp=frame_input.timestamp,
                    detections=detections,
                    inference_time_ms=inference_time,
                    max_threat_level=max_threat,
                    annotated_frame=annotated,
                )

                # Update statistics
                with self._stats_lock:
                    self._stats["frames_processed"] += 1
                    self._stats["total_inference_time_ms"] += inference_time
                    self._stats["detections_count"] += len(detections)

                # Notify result callbacks
                for callback in self._result_callbacks:
                    try:
                        callback(result)
                    except Exception as e:
                        logger.error(f"Result callback error: {e}")

                # Check if alert should be triggered
                if max_threat and max_threat in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                    self._maybe_trigger_alert(result)

            except Exception as e:
                logger.error(f"Worker error: {e}")

    def _maybe_trigger_alert(self, result: DetectionResult):
        """Check cooldown and trigger alert if appropriate."""
        device_id = result.device_id
        current_time = time.time()

        # Check cooldown
        last_alert = self._last_alert_time.get(device_id, 0)
        if current_time - last_alert < self._alert_cooldown_s:
            return

        self._last_alert_time[device_id] = current_time

        # Create alert event
        alert = AlertEvent(
            event_id=str(uuid.uuid4()),
            device_id=device_id,
            timestamp=datetime.fromtimestamp(result.timestamp),
            detections=result.detections,
            max_threat_level=result.max_threat_level,
        )

        with self._stats_lock:
            self._stats["alerts_triggered"] += 1

        logger.warning(
            f"Alert triggered for device {device_id}: "
            f"{len(result.detections)} detections, max threat: {result.max_threat_level.value}"
        )

        # Notify alert callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    def add_result_callback(self, callback: Callable[[DetectionResult], None]):
        """Add a callback for detection results."""
        self._result_callbacks.append(callback)

    def add_alert_callback(self, callback: Callable[[AlertEvent], None]):
        """Add a callback for alerts."""
        self._alert_callbacks.append(callback)

    def set_alert_cooldown(self, cooldown_s: float):
        """Set the alert cooldown period."""
        self._alert_cooldown_s = cooldown_s

    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        with self._stats_lock:
            stats = self._stats.copy()

        stats["queue_size"] = self._frame_queue.qsize()
        stats["avg_inference_time_ms"] = (
            stats["total_inference_time_ms"] / stats["frames_processed"]
            if stats["frames_processed"] > 0 else 0
        )

        return stats

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def detector(self) -> PredatorDetector:
        return self._detector
