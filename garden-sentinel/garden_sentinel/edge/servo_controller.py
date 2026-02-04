"""
Servo controller for pan/tilt camera mount and water gun aiming.
Uses pigpio for hardware PWM on Raspberry Pi.
"""

import logging
import math
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class ServoType(Enum):
    CAMERA_PAN = "camera_pan"
    CAMERA_TILT = "camera_tilt"
    GUN_PAN = "gun_pan"
    GUN_TILT = "gun_tilt"


@dataclass
class ServoConfig:
    # GPIO pins (BCM numbering)
    camera_pan_pin: int = 12
    camera_tilt_pin: int = 13
    gun_pan_pin: int = 18
    gun_tilt_pin: int = 19

    # Servo limits (degrees)
    pan_min: float = -90
    pan_max: float = 90
    tilt_min: float = -45
    tilt_max: float = 45

    # PWM settings
    pwm_frequency: int = 50  # Standard servo frequency
    min_pulse_width: int = 500   # microseconds (0 degrees)
    max_pulse_width: int = 2500  # microseconds (180 degrees)

    # Movement settings
    speed: float = 100  # Degrees per second
    smoothing: bool = True
    smoothing_steps: int = 10

    # Home position (degrees)
    home_pan: float = 0
    home_tilt: float = 0


@dataclass
class ServoState:
    pan: float = 0  # Current pan angle (degrees)
    tilt: float = 0  # Current tilt angle (degrees)
    target_pan: float = 0
    target_tilt: float = 0
    is_moving: bool = False


class ServoController:
    """
    Controls servo motors for camera pan/tilt and water gun aiming.
    Supports smooth movement and tracking.
    """

    def __init__(self, config: ServoConfig, use_gun: bool = True):
        self.config = config
        self.use_gun = use_gun

        self._camera_state = ServoState()
        self._gun_state = ServoState() if use_gun else None

        self._pi = None
        self._mock_mode = False
        self._running = False
        self._movement_thread = None
        self._lock = threading.Lock()

        self._init_gpio()

    def _init_gpio(self):
        """Initialize GPIO with pigpio for hardware PWM."""
        try:
            import pigpio

            self._pi = pigpio.pi()
            if not self._pi.connected:
                raise RuntimeError("Failed to connect to pigpio daemon")

            # Set up PWM for servos
            for pin in [
                self.config.camera_pan_pin,
                self.config.camera_tilt_pin,
            ]:
                self._pi.set_mode(pin, pigpio.OUTPUT)

            if self.use_gun:
                for pin in [
                    self.config.gun_pan_pin,
                    self.config.gun_tilt_pin,
                ]:
                    self._pi.set_mode(pin, pigpio.OUTPUT)

            logger.info("Servo controller initialized with pigpio")

        except ImportError:
            logger.warning("pigpio not available, using mock mode")
            self._mock_mode = True
        except Exception as e:
            logger.warning(f"GPIO not available ({e}), using mock mode")
            self._mock_mode = True

    def start(self):
        """Start the servo controller."""
        self._running = True

        # Go to home position
        self.go_home()

        # Start movement thread
        self._movement_thread = threading.Thread(
            target=self._movement_loop, daemon=True
        )
        self._movement_thread.start()

        logger.info("Servo controller started")

    def stop(self):
        """Stop the servo controller."""
        self._running = False

        if self._movement_thread:
            self._movement_thread.join(timeout=2.0)

        # Go home and disable servos
        self.go_home()
        time.sleep(0.5)

        if self._pi and not self._mock_mode:
            for pin in [
                self.config.camera_pan_pin,
                self.config.camera_tilt_pin,
            ]:
                self._pi.set_servo_pulsewidth(pin, 0)

            if self.use_gun:
                for pin in [self.config.gun_pan_pin, self.config.gun_tilt_pin]:
                    self._pi.set_servo_pulsewidth(pin, 0)

            self._pi.stop()

        logger.info("Servo controller stopped")

    def _angle_to_pulse(self, angle: float, is_tilt: bool = False) -> int:
        """Convert angle to servo pulse width in microseconds."""
        # Normalize angle to 0-180 range for standard servo
        if is_tilt:
            min_angle = self.config.tilt_min
            max_angle = self.config.tilt_max
        else:
            min_angle = self.config.pan_min
            max_angle = self.config.pan_max

        # Clamp angle
        angle = max(min_angle, min(max_angle, angle))

        # Map to 0-180
        normalized = (angle - min_angle) / (max_angle - min_angle) * 180

        # Map to pulse width
        pulse_range = self.config.max_pulse_width - self.config.min_pulse_width
        pulse = self.config.min_pulse_width + (normalized / 180) * pulse_range

        return int(pulse)

    def _set_servo(self, pin: int, pulse_width: int):
        """Set servo position."""
        if self._mock_mode:
            logger.debug(f"[MOCK] Servo pin {pin}: {pulse_width}µs")
            return

        if self._pi:
            self._pi.set_servo_pulsewidth(pin, pulse_width)

    def set_camera_position(self, pan: float, tilt: float, immediate: bool = False):
        """
        Set camera position.

        Args:
            pan: Pan angle in degrees
            tilt: Tilt angle in degrees
            immediate: If True, move immediately without smoothing
        """
        with self._lock:
            self._camera_state.target_pan = max(
                self.config.pan_min, min(self.config.pan_max, pan)
            )
            self._camera_state.target_tilt = max(
                self.config.tilt_min, min(self.config.tilt_max, tilt)
            )

            if immediate:
                self._camera_state.pan = self._camera_state.target_pan
                self._camera_state.tilt = self._camera_state.target_tilt
                self._apply_camera_position()

    def set_gun_position(self, pan: float, tilt: float, immediate: bool = False):
        """Set water gun position."""
        if not self.use_gun or self._gun_state is None:
            return

        with self._lock:
            self._gun_state.target_pan = max(
                self.config.pan_min, min(self.config.pan_max, pan)
            )
            self._gun_state.target_tilt = max(
                self.config.tilt_min, min(self.config.tilt_max, tilt)
            )

            if immediate:
                self._gun_state.pan = self._gun_state.target_pan
                self._gun_state.tilt = self._gun_state.target_tilt
                self._apply_gun_position()

    def track_offset(
        self,
        x_offset: float,
        y_offset: float,
        gain: float = 50,
        track_gun: bool = True,
    ):
        """
        Adjust position based on offset from center.

        Args:
            x_offset: Horizontal offset (-0.5 to 0.5)
            y_offset: Vertical offset (-0.5 to 0.5)
            gain: Sensitivity multiplier (degrees per unit offset)
            track_gun: Also move the water gun
        """
        # Calculate new positions
        pan_adjustment = -x_offset * gain  # Invert X for natural tracking
        tilt_adjustment = y_offset * gain

        with self._lock:
            new_pan = self._camera_state.pan + pan_adjustment
            new_tilt = self._camera_state.tilt + tilt_adjustment

        self.set_camera_position(new_pan, new_tilt)

        if track_gun and self.use_gun:
            self.set_gun_position(new_pan, new_tilt)

    def point_at(self, normalized_x: float, normalized_y: float, track_gun: bool = True):
        """
        Point at a specific location in the frame.

        Args:
            normalized_x: X position (0-1, left to right)
            normalized_y: Y position (0-1, top to bottom)
            track_gun: Also point the water gun
        """
        # Convert to offsets from center
        x_offset = normalized_x - 0.5
        y_offset = normalized_y - 0.5

        # Convert to angles (assuming FOV of ~60 degrees)
        fov_h = 60
        fov_v = 45

        pan = x_offset * fov_h
        tilt = y_offset * fov_v

        self.set_camera_position(pan, tilt)

        if track_gun and self.use_gun:
            self.set_gun_position(pan, tilt)

    def go_home(self):
        """Return to home position."""
        self.set_camera_position(
            self.config.home_pan,
            self.config.home_tilt,
            immediate=True,
        )
        if self.use_gun:
            self.set_gun_position(
                self.config.home_pan,
                self.config.home_tilt,
                immediate=True,
            )

    def _apply_camera_position(self):
        """Apply current camera position to servos."""
        pan_pulse = self._angle_to_pulse(self._camera_state.pan, is_tilt=False)
        tilt_pulse = self._angle_to_pulse(self._camera_state.tilt, is_tilt=True)

        self._set_servo(self.config.camera_pan_pin, pan_pulse)
        self._set_servo(self.config.camera_tilt_pin, tilt_pulse)

    def _apply_gun_position(self):
        """Apply current gun position to servos."""
        if not self.use_gun or self._gun_state is None:
            return

        pan_pulse = self._angle_to_pulse(self._gun_state.pan, is_tilt=False)
        tilt_pulse = self._angle_to_pulse(self._gun_state.tilt, is_tilt=True)

        self._set_servo(self.config.gun_pan_pin, pan_pulse)
        self._set_servo(self.config.gun_tilt_pin, tilt_pulse)

    def _movement_loop(self):
        """Background thread for smooth servo movement."""
        update_rate = 50  # Hz
        interval = 1.0 / update_rate
        max_step = self.config.speed / update_rate

        while self._running:
            try:
                with self._lock:
                    # Update camera position
                    camera_moved = self._update_position(
                        self._camera_state, max_step
                    )
                    if camera_moved:
                        self._apply_camera_position()

                    # Update gun position
                    if self.use_gun and self._gun_state:
                        gun_moved = self._update_position(
                            self._gun_state, max_step
                        )
                        if gun_moved:
                            self._apply_gun_position()

                time.sleep(interval)

            except Exception as e:
                logger.error(f"Movement loop error: {e}")
                time.sleep(0.1)

    def _update_position(self, state: ServoState, max_step: float) -> bool:
        """
        Update position towards target with speed limiting.

        Returns:
            True if position changed
        """
        pan_diff = state.target_pan - state.pan
        tilt_diff = state.target_tilt - state.tilt

        if abs(pan_diff) < 0.1 and abs(tilt_diff) < 0.1:
            state.is_moving = False
            return False

        state.is_moving = True

        # Limit step size
        if abs(pan_diff) > max_step:
            pan_diff = max_step if pan_diff > 0 else -max_step
        if abs(tilt_diff) > max_step:
            tilt_diff = max_step if tilt_diff > 0 else -max_step

        state.pan += pan_diff
        state.tilt += tilt_diff

        return True

    def get_camera_position(self) -> Tuple[float, float]:
        """Get current camera position (pan, tilt) in degrees."""
        with self._lock:
            return (self._camera_state.pan, self._camera_state.tilt)

    def get_gun_position(self) -> Optional[Tuple[float, float]]:
        """Get current gun position (pan, tilt) in degrees."""
        if not self.use_gun or self._gun_state is None:
            return None
        with self._lock:
            return (self._gun_state.pan, self._gun_state.tilt)

    @property
    def is_mock(self) -> bool:
        return self._mock_mode
