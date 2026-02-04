"""
Video recorder for capturing clips around detection events.
Maintains a rolling buffer and saves clips when triggered.
"""

import logging
import os
import queue
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RecorderConfig:
    enabled: bool = True
    # Rolling buffer settings
    pre_event_seconds: float = 5.0      # Seconds to keep before event
    post_event_seconds: float = 10.0    # Seconds to record after event
    # Output settings
    output_dir: str = "/var/lib/garden-sentinel/recordings"
    max_storage_mb: int = 5000          # Max storage before cleanup
    video_format: str = "mp4"           # mp4 or avi
    video_codec: str = "mp4v"           # mp4v, XVID, H264
    fps: int = 15                       # Recording FPS
    # Quality
    jpeg_quality: int = 85
    resolution_scale: float = 1.0       # Scale factor for recording resolution


class RollingBuffer:
    """
    Thread-safe rolling buffer for video frames.
    Keeps the last N seconds of frames.
    """

    def __init__(self, max_seconds: float, fps: int):
        self.max_frames = int(max_seconds * fps)
        self._buffer: deque = deque(maxlen=self.max_frames)
        self._lock = threading.Lock()

    def add(self, frame: np.ndarray, timestamp: float):
        """Add a frame to the buffer."""
        with self._lock:
            self._buffer.append((frame.copy(), timestamp))

    def get_all(self) -> list[tuple[np.ndarray, float]]:
        """Get all frames in the buffer."""
        with self._lock:
            return list(self._buffer)

    def get_since(self, since_timestamp: float) -> list[tuple[np.ndarray, float]]:
        """Get frames since a specific timestamp."""
        with self._lock:
            return [(f, t) for f, t in self._buffer if t >= since_timestamp]

    def clear(self):
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()

    @property
    def frame_count(self) -> int:
        with self._lock:
            return len(self._buffer)


class EventRecorder:
    """
    Records video clips around detection events.

    Flow:
    1. Continuously buffers recent frames (rolling buffer)
    2. When triggered, saves pre-event buffer
    3. Continues recording for post-event duration
    4. Saves complete clip with metadata
    """

    def __init__(self, config: RecorderConfig, device_id: str):
        self.config = config
        self.device_id = device_id

        self._rolling_buffer = RollingBuffer(
            config.pre_event_seconds,
            config.fps
        )

        self._is_recording = False
        self._recording_start_time: float = 0
        self._recording_event_id: str = ""
        self._recording_frames: list = []
        self._recording_metadata: dict = {}

        self._output_dir = Path(config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._running = False
        self._save_queue: queue.Queue = queue.Queue()
        self._save_thread: Optional[threading.Thread] = None

    def start(self):
        """Start the recorder."""
        if not self.config.enabled:
            return

        self._running = True
        self._save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self._save_thread.start()
        logger.info("Event recorder started")

    def stop(self):
        """Stop the recorder."""
        self._running = False
        if self._save_thread:
            self._save_queue.put(None)  # Signal to stop
            self._save_thread.join(timeout=5.0)

    def add_frame(self, frame: np.ndarray, timestamp: float):
        """
        Add a frame to the rolling buffer or current recording.
        Should be called for every frame.
        """
        if not self.config.enabled:
            return

        # Scale frame if needed
        if self.config.resolution_scale != 1.0:
            h, w = frame.shape[:2]
            new_w = int(w * self.config.resolution_scale)
            new_h = int(h * self.config.resolution_scale)
            frame = cv2.resize(frame, (new_w, new_h))

        with self._lock:
            if self._is_recording:
                # Add to active recording
                self._recording_frames.append((frame.copy(), timestamp))

                # Check if recording duration exceeded
                elapsed = timestamp - self._recording_start_time
                if elapsed >= self.config.pre_event_seconds + self.config.post_event_seconds:
                    self._finish_recording()
            else:
                # Add to rolling buffer
                self._rolling_buffer.add(frame, timestamp)

    def trigger_recording(
        self,
        event_id: str,
        trigger_time: float,
        metadata: Optional[dict] = None,
    ):
        """
        Trigger recording of an event.

        Args:
            event_id: Unique identifier for this event
            trigger_time: Timestamp when the event occurred
            metadata: Additional metadata to save with the recording
        """
        if not self.config.enabled:
            return

        with self._lock:
            if self._is_recording:
                logger.warning("Already recording, ignoring trigger")
                return

            logger.info(f"Starting event recording: {event_id}")

            # Get pre-event frames from buffer
            pre_event_start = trigger_time - self.config.pre_event_seconds
            pre_frames = self._rolling_buffer.get_since(pre_event_start)

            self._is_recording = True
            self._recording_start_time = trigger_time - self.config.pre_event_seconds
            self._recording_event_id = event_id
            self._recording_frames = pre_frames.copy()
            self._recording_metadata = metadata or {}
            self._recording_metadata["event_id"] = event_id
            self._recording_metadata["trigger_time"] = trigger_time
            self._recording_metadata["device_id"] = self.device_id

            # Clear rolling buffer since we've captured it
            self._rolling_buffer.clear()

    def _finish_recording(self):
        """Finish current recording and queue for saving."""
        if not self._is_recording:
            return

        event_id = self._recording_event_id
        frames = self._recording_frames.copy()
        metadata = self._recording_metadata.copy()

        self._is_recording = False
        self._recording_frames = []
        self._recording_metadata = {}
        self._recording_event_id = ""

        # Queue for async saving
        self._save_queue.put((event_id, frames, metadata))
        logger.info(f"Recording queued for saving: {event_id} ({len(frames)} frames)")

    def _save_worker(self):
        """Background worker for saving recordings."""
        while self._running:
            try:
                item = self._save_queue.get(timeout=1.0)
                if item is None:
                    break

                event_id, frames, metadata = item
                self._save_recording(event_id, frames, metadata)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Save worker error: {e}")

    def _save_recording(
        self,
        event_id: str,
        frames: list[tuple[np.ndarray, float]],
        metadata: dict,
    ):
        """Save a recording to disk."""
        if not frames:
            logger.warning(f"No frames to save for {event_id}")
            return

        # Create output directory for this event
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        event_dir = self._output_dir / f"{timestamp}_{event_id}"
        event_dir.mkdir(parents=True, exist_ok=True)

        # Determine video dimensions from first frame
        first_frame = frames[0][0]
        height, width = first_frame.shape[:2]

        # Create video writer
        video_path = event_dir / f"recording.{self.config.video_format}"
        fourcc = cv2.VideoWriter_fourcc(*self.config.video_codec)
        writer = cv2.VideoWriter(
            str(video_path),
            fourcc,
            self.config.fps,
            (width, height),
        )

        if not writer.isOpened():
            logger.error(f"Failed to create video writer for {video_path}")
            return

        try:
            # Write frames
            for frame, _ in frames:
                writer.write(frame)

            writer.release()
            logger.info(f"Saved recording: {video_path}")

            # Save thumbnail (frame at trigger time)
            trigger_time = metadata.get("trigger_time", 0)
            thumb_frame = None
            for frame, ts in frames:
                if ts >= trigger_time:
                    thumb_frame = frame
                    break

            if thumb_frame is not None:
                thumb_path = event_dir / "thumbnail.jpg"
                cv2.imwrite(str(thumb_path), thumb_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

            # Save metadata
            metadata["video_file"] = str(video_path)
            metadata["frame_count"] = len(frames)
            metadata["duration_seconds"] = len(frames) / self.config.fps
            metadata["resolution"] = {"width": width, "height": height}
            metadata["saved_at"] = datetime.now().isoformat()

            import json
            metadata_path = event_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

            # Check storage limits
            self._cleanup_old_recordings()

        except Exception as e:
            logger.error(f"Error saving recording: {e}")
            writer.release()

    def _cleanup_old_recordings(self):
        """Remove old recordings if storage limit exceeded."""
        try:
            # Calculate total storage used
            total_size = sum(
                f.stat().st_size
                for f in self._output_dir.rglob("*")
                if f.is_file()
            )
            total_mb = total_size / (1024 * 1024)

            if total_mb <= self.config.max_storage_mb:
                return

            # Get recordings sorted by date (oldest first)
            recordings = sorted(
                [d for d in self._output_dir.iterdir() if d.is_dir()],
                key=lambda d: d.stat().st_mtime,
            )

            # Remove oldest until under limit
            for recording_dir in recordings:
                if total_mb <= self.config.max_storage_mb * 0.8:  # Keep 20% buffer
                    break

                dir_size = sum(
                    f.stat().st_size
                    for f in recording_dir.rglob("*")
                    if f.is_file()
                )

                shutil.rmtree(recording_dir)
                total_mb -= dir_size / (1024 * 1024)
                logger.info(f"Removed old recording: {recording_dir.name}")

        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    def cancel_recording(self):
        """Cancel current recording without saving."""
        with self._lock:
            if self._is_recording:
                logger.info(f"Cancelled recording: {self._recording_event_id}")
                self._is_recording = False
                self._recording_frames = []
                self._recording_metadata = {}
                self._recording_event_id = ""

    def get_recordings(self, limit: int = 20) -> list[dict]:
        """Get list of saved recordings."""
        recordings = []

        for recording_dir in sorted(
            self._output_dir.iterdir(),
            key=lambda d: d.stat().st_mtime if d.is_dir() else 0,
            reverse=True,
        )[:limit]:
            if not recording_dir.is_dir():
                continue

            metadata_path = recording_dir / "metadata.json"
            if metadata_path.exists():
                import json
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    metadata["path"] = str(recording_dir)
                    recordings.append(metadata)

        return recordings

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def buffer_frame_count(self) -> int:
        return self._rolling_buffer.frame_count
