"""
GPIO controller for actuators (alarm, water sprayer, LEDs).
Uses gpiozero for Raspberry Pi GPIO control.
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GPIOConfig:
    alarm_pin: int = 17
    sprayer_pin: int = 27
    status_led_pin: int = 22
    ir_led_pin: int = 23
    alarm_duration_s: float = 10
    sprayer_duration_s: float = 5
    sprayer_cooldown_s: float = 30


class GPIOController:
    """
    Controls GPIO pins for actuators.
    Uses gpiozero on Raspberry Pi, falls back to mock for testing.
    """

    def __init__(self, config: GPIOConfig):
        self.config = config
        self._alarm = None
        self._sprayer = None
        self._status_led = None
        self._ir_led = None
        self._mock_mode = False

        self._alarm_timer: Optional[threading.Timer] = None
        self._sprayer_timer: Optional[threading.Timer] = None
        self._last_spray_time: float = 0

        self._init_gpio()

    def _init_gpio(self):
        """Initialize GPIO pins."""
        try:
            from gpiozero import LED, OutputDevice

            self._alarm = OutputDevice(self.config.alarm_pin, active_high=True, initial_value=False)
            self._sprayer = OutputDevice(self.config.sprayer_pin, active_high=True, initial_value=False)
            self._status_led = LED(self.config.status_led_pin)
            self._ir_led = LED(self.config.ir_led_pin)

            logger.info("GPIO initialized with gpiozero")

        except (ImportError, Exception) as e:
            logger.warning(f"GPIO not available ({e}), using mock mode")
            self._mock_mode = True
            self._mock_states = {
                "alarm": False,
                "sprayer": False,
                "status_led": False,
                "ir_led": False,
            }

    def activate_alarm(self, duration_s: Optional[float] = None):
        """
        Activate the alarm for a specified duration.
        If duration is None, uses the default from config.
        """
        if duration_s is None:
            duration_s = self.config.alarm_duration_s

        # Cancel any existing timer
        if self._alarm_timer:
            self._alarm_timer.cancel()

        if self._mock_mode:
            self._mock_states["alarm"] = True
            logger.info(f"[MOCK] Alarm ACTIVATED for {duration_s}s")
        else:
            self._alarm.on()
            logger.info(f"Alarm ACTIVATED for {duration_s}s")

        # Set timer to deactivate
        self._alarm_timer = threading.Timer(duration_s, self.deactivate_alarm)
        self._alarm_timer.start()

    def deactivate_alarm(self):
        """Deactivate the alarm."""
        if self._alarm_timer:
            self._alarm_timer.cancel()
            self._alarm_timer = None

        if self._mock_mode:
            self._mock_states["alarm"] = False
            logger.info("[MOCK] Alarm DEACTIVATED")
        else:
            self._alarm.off()
            logger.info("Alarm DEACTIVATED")

    def activate_sprayer(self, duration_s: Optional[float] = None) -> bool:
        """
        Activate the water sprayer for a specified duration.
        Returns False if still in cooldown period.
        """
        if duration_s is None:
            duration_s = self.config.sprayer_duration_s

        # Check cooldown
        time_since_last = time.time() - self._last_spray_time
        if time_since_last < self.config.sprayer_cooldown_s:
            remaining = self.config.sprayer_cooldown_s - time_since_last
            logger.warning(f"Sprayer in cooldown, {remaining:.1f}s remaining")
            return False

        # Cancel any existing timer
        if self._sprayer_timer:
            self._sprayer_timer.cancel()

        self._last_spray_time = time.time()

        if self._mock_mode:
            self._mock_states["sprayer"] = True
            logger.info(f"[MOCK] Sprayer ACTIVATED for {duration_s}s")
        else:
            self._sprayer.on()
            logger.info(f"Sprayer ACTIVATED for {duration_s}s")

        # Set timer to deactivate
        self._sprayer_timer = threading.Timer(duration_s, self.deactivate_sprayer)
        self._sprayer_timer.start()
        return True

    def deactivate_sprayer(self):
        """Deactivate the water sprayer."""
        if self._sprayer_timer:
            self._sprayer_timer.cancel()
            self._sprayer_timer = None

        if self._mock_mode:
            self._mock_states["sprayer"] = False
            logger.info("[MOCK] Sprayer DEACTIVATED")
        else:
            self._sprayer.off()
            logger.info("Sprayer DEACTIVATED")

    def set_status_led(self, on: bool):
        """Set the status LED state."""
        if self._mock_mode:
            self._mock_states["status_led"] = on
            logger.debug(f"[MOCK] Status LED: {'ON' if on else 'OFF'}")
        else:
            if on:
                self._status_led.on()
            else:
                self._status_led.off()

    def blink_status_led(self, on_time: float = 0.5, off_time: float = 0.5, n: int = 3):
        """Blink the status LED."""
        if self._mock_mode:
            logger.debug(f"[MOCK] Status LED blinking {n} times")
        else:
            self._status_led.blink(on_time=on_time, off_time=off_time, n=n)

    def set_ir_led(self, on: bool):
        """Set the IR LED state (for night vision)."""
        if self._mock_mode:
            self._mock_states["ir_led"] = on
            logger.debug(f"[MOCK] IR LED: {'ON' if on else 'OFF'}")
        else:
            if on:
                self._ir_led.on()
            else:
                self._ir_led.off()

    def get_states(self) -> dict:
        """Get the current state of all actuators."""
        if self._mock_mode:
            return self._mock_states.copy()
        else:
            return {
                "alarm": self._alarm.is_active if self._alarm else False,
                "sprayer": self._sprayer.is_active if self._sprayer else False,
                "status_led": self._status_led.is_lit if self._status_led else False,
                "ir_led": self._ir_led.is_lit if self._ir_led else False,
            }

    def cleanup(self):
        """Cleanup GPIO resources."""
        # Cancel any timers
        if self._alarm_timer:
            self._alarm_timer.cancel()
        if self._sprayer_timer:
            self._sprayer_timer.cancel()

        # Turn everything off
        self.deactivate_alarm()
        self.deactivate_sprayer()
        self.set_status_led(False)
        self.set_ir_led(False)

        if not self._mock_mode:
            try:
                if self._alarm:
                    self._alarm.close()
                if self._sprayer:
                    self._sprayer.close()
                if self._status_led:
                    self._status_led.close()
                if self._ir_led:
                    self._ir_led.close()
            except Exception as e:
                logger.warning(f"Error cleaning up GPIO: {e}")

        logger.info("GPIO cleanup complete")

    @property
    def is_mock(self) -> bool:
        """Check if running in mock mode."""
        return self._mock_mode

    @property
    def sprayer_cooldown_remaining(self) -> float:
        """Get remaining cooldown time for sprayer in seconds."""
        time_since_last = time.time() - self._last_spray_time
        remaining = self.config.sprayer_cooldown_s - time_since_last
        return max(0, remaining)
