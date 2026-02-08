"""
Targeting controller for coordinating detection, tracking, and deterrent aiming.
Handles the complete pipeline from detection to water spray.
"""

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Tuple

import numpy as np

from garden_sentinel.shared import Detection, ThreatLevel

logger = logging.getLogger(__name__)


class TargetingState(Enum):
    IDLE = "idle"
    ACQUIRING = "acquiring"      # Got detection, initializing tracker
    TRACKING = "tracking"        # Actively tracking target
    ENGAGING = "engaging"        # Aiming and preparing to spray
    SPRAYING = "spraying"        # Currently spraying
    LOST = "lost"                # Lost target, searching
    COOLDOWN = "cooldown"        # Post-spray cooldown


@dataclass
class TargetingConfig:
    # Threat level threshold to engage
    min_threat_level: ThreatLevel = ThreatLevel.HIGH

    # Tracking settings
    acquire_timeout_s: float = 2.0      # Max time to acquire target
    track_timeout_s: float = 5.0        # Max time tracking without confirmation
    lost_timeout_s: float = 3.0         # Time to search before giving up

    # Aiming settings
    aim_tolerance: float = 0.05         # How close to center before engaging (normalized)
    lead_factor: float = 0.3            # How much to lead moving targets
    aim_smoothing: float = 0.8          # Smoothing for aim adjustments

    # Spray settings
    spray_duration_s: float = 3.0       # How long to spray
    spray_cooldown_s: float = 10.0      # Cooldown between sprays
    confirm_aim_frames: int = 5         # Frames on target before spraying

    # Servo settings
    servo_speed: float = 100            # Degrees per second
    pan_limits: Tuple[float, float] = (-90, 90)
    tilt_limits: Tuple[float, float] = (-45, 45)


@dataclass
class Target:
    detection: Detection
    first_seen: float
    last_seen: float
    position: Tuple[float, float]       # Normalized (x, y)
    velocity: Tuple[float, float] = (0.0, 0.0)  # Normalized velocity
    frames_tracked: int = 0
    frames_on_aim: int = 0              # Frames where target is centered


class TargetingController:
    """
    Coordinates detection, tracking, and deterrent systems.

    Flow:
    1. Receives detection from server or edge inference
    2. Evaluates threat level and decides to engage
    3. Initializes tracker on target
    4. Tracks target and controls servos to aim
    5. Predicts target movement and leads aim
    6. Activates sprayer when on target
    """

    def __init__(
        self,
        config: TargetingConfig,
        servo_controller=None,
        gpio_controller=None,
        tracker=None,
    ):
        self.config = config
        self.servo = servo_controller
        self.gpio = gpio_controller
        self.tracker = tracker

        self._state = TargetingState.IDLE
        self._target: Optional[Target] = None
        self._state_start_time: float = 0

        self._position_history: list[Tuple[float, float, float]] = []  # (x, y, time)
        self._aim_position: Tuple[float, float] = (0.5, 0.5)  # Current aim point

        self._lock = threading.Lock()
        self._running = False
        self._control_thread: Optional[threading.Thread] = None

        # Callbacks
        self._on_state_change: Optional[Callable[[TargetingState, Optional[Target]], None]] = None
        self._on_engage: Optional[Callable[[Target], None]] = None

    def start(self):
        """Start the targeting controller."""
        self._running = True
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()
        logger.info("Targeting controller started")

    def stop(self):
        """Stop the targeting controller."""
        self._running = False
        if self._control_thread:
            self._control_thread.join(timeout=2.0)
        self._set_state(TargetingState.IDLE)
        logger.info("Targeting controller stopped")

    def process_detections(self, detections: list[Detection], frame=None, timestamp: float = None):
        """
        Process new detections and decide whether to engage.

        Args:
            detections: List of detections from server or edge
            frame: Optional frame for tracker initialization
            timestamp: Detection timestamp
        """
        if timestamp is None:
            timestamp = time.time()

        with self._lock:
            # Filter for threatening detections
            threats = [
                d for d in detections
                if d.threat_level and self._threat_level_value(d.threat_level) >=
                   self._threat_level_value(self.config.min_threat_level)
            ]

            if not threats:
                return

            # Sort by threat level (highest first)
            threats.sort(key=lambda d: self._threat_level_value(d.threat_level), reverse=True)
            highest_threat = threats[0]

            # Calculate center position
            center_x = highest_threat.bbox.x + highest_threat.bbox.width / 2
            center_y = highest_threat.bbox.y + highest_threat.bbox.height / 2

            if self._state == TargetingState.IDLE:
                # New target - start acquiring
                self._target = Target(
                    detection=highest_threat,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    position=(center_x, center_y),
                )
                self._set_state(TargetingState.ACQUIRING)

                # Initialize tracker if available
                if self.tracker and frame is not None:
                    bbox = (
                        highest_threat.bbox.x,
                        highest_threat.bbox.y,
                        highest_threat.bbox.width,
                        highest_threat.bbox.height,
                    )
                    self.tracker.start_tracking(frame, bbox, highest_threat.class_name)

            elif self._state in [TargetingState.TRACKING, TargetingState.ENGAGING, TargetingState.LOST]:
                # Update existing target
                if self._target:
                    self._update_target_position(center_x, center_y, timestamp)
                    self._target.last_seen = timestamp
                    self._target.detection = highest_threat

                    # Reinforce tracker
                    if self.tracker:
                        bbox = (
                            highest_threat.bbox.x,
                            highest_threat.bbox.y,
                            highest_threat.bbox.width,
                            highest_threat.bbox.height,
                        )
                        self.tracker.reinforce(bbox, highest_threat.confidence)

                    # If we were lost, resume tracking
                    if self._state == TargetingState.LOST:
                        self._set_state(TargetingState.TRACKING)

    def process_tracker_update(self, tracker_state, timestamp: float = None):
        """
        Process update from the object tracker.
        Called when no server detection but tracker is still tracking.
        """
        if timestamp is None:
            timestamp = time.time()

        with self._lock:
            if not tracker_state.is_tracking or self._target is None:
                return

            if tracker_state.center:
                self._update_target_position(
                    tracker_state.center[0],
                    tracker_state.center[1],
                    timestamp,
                )
                self._target.frames_tracked += 1

    def _update_target_position(self, x: float, y: float, timestamp: float):
        """Update target position and calculate velocity."""
        if self._target is None:
            return

        old_pos = self._target.position
        self._target.position = (x, y)

        # Add to history
        self._position_history.append((x, y, timestamp))

        # Keep only recent history (last 1 second)
        cutoff = timestamp - 1.0
        self._position_history = [
            p for p in self._position_history if p[2] > cutoff
        ]

        # Calculate velocity from history
        if len(self._position_history) >= 2:
            oldest = self._position_history[0]
            newest = self._position_history[-1]
            dt = newest[2] - oldest[2]

            if dt > 0.05:  # At least 50ms of data
                vx = (newest[0] - oldest[0]) / dt
                vy = (newest[1] - oldest[1]) / dt

                # Smooth velocity
                alpha = 0.7
                old_vx, old_vy = self._target.velocity
                self._target.velocity = (
                    alpha * old_vx + (1 - alpha) * vx,
                    alpha * old_vy + (1 - alpha) * vy,
                )

    def _control_loop(self):
        """Main control loop running in background thread."""
        update_rate = 30  # Hz
        interval = 1.0 / update_rate

        while self._running:
            try:
                with self._lock:
                    self._update_state_machine()
                    self._update_aim()

                time.sleep(interval)

            except Exception as e:
                logger.error(f"Control loop error: {e}")
                time.sleep(0.1)

    def _update_state_machine(self):
        """Update the targeting state machine."""
        now = time.time()
        elapsed = now - self._state_start_time

        if self._state == TargetingState.ACQUIRING:
            # Check if we have good track
            if self._target and self._target.frames_tracked >= 3:
                self._set_state(TargetingState.TRACKING)
            elif elapsed > self.config.acquire_timeout_s:
                logger.warning("Failed to acquire target")
                self._set_state(TargetingState.IDLE)
                self._target = None

        elif self._state == TargetingState.TRACKING:
            # Check if we should engage
            if self._target and self._is_on_target():
                self._target.frames_on_aim += 1
                if self._target.frames_on_aim >= self.config.confirm_aim_frames:
                    self._set_state(TargetingState.ENGAGING)
            else:
                if self._target:
                    self._target.frames_on_aim = 0

            # Check for track loss
            if self._target:
                time_since_seen = now - self._target.last_seen
                if time_since_seen > 0.5:  # No update for 500ms
                    self._set_state(TargetingState.LOST)

        elif self._state == TargetingState.ENGAGING:
            # Activate sprayer
            if self.gpio:
                success = self.gpio.activate_sprayer(self.config.spray_duration_s)
                if success:
                    self._set_state(TargetingState.SPRAYING)
                    if self._on_engage and self._target:
                        self._on_engage(self._target)
                else:
                    # Sprayer in cooldown, go back to tracking
                    self._set_state(TargetingState.TRACKING)
            else:
                # No GPIO, simulate spray
                logger.info("[SIMULATED] Spraying target!")
                self._set_state(TargetingState.SPRAYING)

        elif self._state == TargetingState.SPRAYING:
            if elapsed >= self.config.spray_duration_s:
                self._set_state(TargetingState.COOLDOWN)

        elif self._state == TargetingState.COOLDOWN:
            if elapsed >= self.config.spray_cooldown_s:
                self._set_state(TargetingState.IDLE)
                self._target = None
                self._position_history.clear()

        elif self._state == TargetingState.LOST:
            if elapsed >= self.config.lost_timeout_s:
                logger.info("Target lost, returning to idle")
                self._set_state(TargetingState.IDLE)
                self._target = None
                self._position_history.clear()

    def _update_aim(self):
        """Update servo aim position based on target."""
        if self._target is None or self._state == TargetingState.IDLE:
            return

        # Get current target position
        target_x, target_y = self._target.position

        # Lead the target based on velocity
        if self._state in [TargetingState.TRACKING, TargetingState.ENGAGING]:
            vx, vy = self._target.velocity
            lead_time = self.config.lead_factor
            target_x += vx * lead_time
            target_y += vy * lead_time

        # Clamp to valid range
        target_x = max(0, min(1, target_x))
        target_y = max(0, min(1, target_y))

        # Smooth the aim
        alpha = self.config.aim_smoothing
        self._aim_position = (
            alpha * self._aim_position[0] + (1 - alpha) * target_x,
            alpha * self._aim_position[1] + (1 - alpha) * target_y,
        )

        # Send to servo controller
        if self.servo:
            self.servo.point_at(
                self._aim_position[0],
                self._aim_position[1],
                track_gun=True,
            )

    def _is_on_target(self) -> bool:
        """Check if aim is on target within tolerance."""
        if self._target is None:
            return False

        target_x, target_y = self._target.position
        aim_x, aim_y = self._aim_position

        distance = math.sqrt((target_x - aim_x) ** 2 + (target_y - aim_y) ** 2)
        return distance <= self.config.aim_tolerance

    def _set_state(self, new_state: TargetingState):
        """Change state and notify."""
        if new_state != self._state:
            old_state = self._state
            self._state = new_state
            self._state_start_time = time.time()

            logger.info(f"Targeting state: {old_state.value} -> {new_state.value}")

            if self._on_state_change:
                self._on_state_change(new_state, self._target)

    def _threat_level_value(self, level: ThreatLevel) -> int:
        """Convert threat level to numeric value for comparison."""
        order = [ThreatLevel.LOW, ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL]
        return order.index(level) if level in order else 0

    def set_state_change_callback(self, callback: Callable[[TargetingState, Optional[Target]], None]):
        """Set callback for state changes."""
        self._on_state_change = callback

    def set_engage_callback(self, callback: Callable[[Target], None]):
        """Set callback when target is engaged (sprayed)."""
        self._on_engage = callback

    def disengage(self):
        """Manually disengage and return to idle."""
        with self._lock:
            if self.gpio:
                self.gpio.deactivate_sprayer()
            self._set_state(TargetingState.IDLE)
            self._target = None

    def get_status(self) -> dict:
        """Get current targeting status."""
        with self._lock:
            return {
                "state": self._state.value,
                "has_target": self._target is not None,
                "target_class": self._target.detection.class_name if self._target else None,
                "target_position": self._target.position if self._target else None,
                "aim_position": self._aim_position,
                "frames_tracked": self._target.frames_tracked if self._target else 0,
                "frames_on_aim": self._target.frames_on_aim if self._target else 0,
            }

    @property
    def state(self) -> TargetingState:
        return self._state

    @property
    def is_engaged(self) -> bool:
        return self._state in [TargetingState.ENGAGING, TargetingState.SPRAYING]
