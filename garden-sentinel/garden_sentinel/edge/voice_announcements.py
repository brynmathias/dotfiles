"""
Voice announcement system for Garden Sentinel.

Provides spoken alerts and status updates for accessibility
and hands-free operation.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable
from enum import Enum
import asyncio
import logging
import time
import queue
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class AnnouncementPriority(Enum):
    """Priority levels for announcements."""
    LOW = 1      # Status updates, non-urgent info
    NORMAL = 2   # Standard alerts
    HIGH = 3     # Important alerts
    CRITICAL = 4 # Immediate threats


class AnnouncementType(Enum):
    """Types of announcements."""
    ALERT = "alert"
    STATUS = "status"
    WEATHER = "weather"
    SYSTEM = "system"
    CUSTOM = "custom"


@dataclass
class Announcement:
    """An announcement to be spoken."""
    id: str
    text: str
    priority: AnnouncementPriority = AnnouncementPriority.NORMAL
    announcement_type: AnnouncementType = AnnouncementType.ALERT
    timestamp: float = field(default_factory=time.time)
    repeat_count: int = 1
    repeat_delay: float = 3.0  # seconds between repeats
    interruptible: bool = True
    expires_at: Optional[float] = None  # Don't play if expired

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


@dataclass
class VoiceConfig:
    """Configuration for voice synthesis."""
    engine: str = "pyttsx3"  # pyttsx3, espeak, google, aws
    voice_id: Optional[str] = None
    rate: int = 150  # words per minute
    volume: float = 0.9  # 0.0 to 1.0
    pitch: int = 50  # 0 to 100 (engine dependent)
    language: str = "en"


class TTSEngine:
    """Text-to-speech engine abstraction."""

    def __init__(self, config: VoiceConfig):
        self.config = config
        self._engine = None
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize the TTS engine."""
        try:
            if self.config.engine == "pyttsx3":
                return self._init_pyttsx3()
            elif self.config.engine == "espeak":
                return self._init_espeak()
            else:
                logger.warning(f"Unknown TTS engine: {self.config.engine}")
                return self._init_pyttsx3()  # Fallback

        except Exception as e:
            logger.error(f"Failed to initialize TTS: {e}")
            return False

    def _init_pyttsx3(self) -> bool:
        """Initialize pyttsx3 engine."""
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self.config.rate)
            self._engine.setProperty("volume", self.config.volume)

            if self.config.voice_id:
                self._engine.setProperty("voice", self.config.voice_id)

            self._initialized = True
            logger.info("Initialized pyttsx3 TTS engine")
            return True

        except ImportError:
            logger.warning("pyttsx3 not available")
            return False

    def _init_espeak(self) -> bool:
        """Initialize espeak (via subprocess)."""
        import subprocess
        try:
            result = subprocess.run(
                ["espeak", "--version"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                self._engine = "espeak"
                self._initialized = True
                logger.info("Initialized espeak TTS engine")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return False

    def speak(self, text: str) -> bool:
        """Speak the given text."""
        if not self._initialized:
            if not self.initialize():
                logger.error("TTS not initialized")
                return False

        try:
            if self.config.engine == "pyttsx3" and self._engine:
                self._engine.say(text)
                self._engine.runAndWait()
                return True

            elif self.config.engine == "espeak":
                import subprocess
                args = ["espeak"]
                args.extend(["-s", str(self.config.rate)])
                args.extend(["-a", str(int(self.config.volume * 200))])
                if self.config.voice_id:
                    args.extend(["-v", self.config.voice_id])
                args.append(text)

                subprocess.run(args, timeout=30)
                return True

            return False

        except Exception as e:
            logger.error(f"TTS speak failed: {e}")
            return False

    def get_available_voices(self) -> List[Dict[str, str]]:
        """Get list of available voices."""
        if self.config.engine == "pyttsx3" and self._engine:
            voices = self._engine.getProperty("voices")
            return [
                {"id": v.id, "name": v.name, "languages": v.languages}
                for v in voices
            ]
        return []


class VoiceAnnouncementSystem:
    """
    Voice announcement system for Garden Sentinel.

    Features:
    - Priority-based announcement queue
    - Multiple TTS engine support
    - Quiet hours support
    - Announcement templates
    - Rate limiting
    """

    def __init__(
        self,
        config: Optional[VoiceConfig] = None,
        storage_path: Optional[Path] = None,
    ):
        self.config = config or VoiceConfig()
        self.storage_path = storage_path
        self.tts = TTSEngine(self.config)

        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._current_announcement: Optional[Announcement] = None

        # Rate limiting
        self._last_announcement_time: Dict[str, float] = {}
        self._min_interval = 5.0  # Minimum seconds between same-type announcements

        # Quiet hours
        self.quiet_hours_enabled = False
        self.quiet_hours_start = 22  # 10 PM
        self.quiet_hours_end = 7     # 7 AM

        # Templates
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load announcement templates."""
        return {
            "predator_detected": "Alert! {predator_type} detected in {location}.",
            "predator_detected_high": "Warning! {predator_type} detected with high confidence in {location}. Activating deterrents.",
            "multiple_predators": "Multiple threat alert! {count} predators detected.",
            "deterrence_activated": "Activating {method} deterrent.",
            "deterrence_success": "Threat has been deterred successfully.",
            "device_offline": "Warning: Camera {device_name} has gone offline.",
            "device_online": "Camera {device_name} is back online.",
            "low_battery": "Warning: Camera {device_name} has low battery at {percent} percent.",
            "system_armed": "Garden Sentinel system is now armed and monitoring.",
            "system_disarmed": "Garden Sentinel system is now disarmed.",
            "weather_alert": "Weather alert: {condition}. Detection sensitivity adjusted.",
            "quiet_hours_start": "Entering quiet hours. Voice announcements reduced.",
            "quiet_hours_end": "Quiet hours ended. Normal announcements resumed.",
            "daily_summary": "Daily summary: {detections} detections, {deterrence_rate} percent deterrence success rate.",
        }

    def start(self):
        """Start the announcement system."""
        if self._running:
            return

        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        logger.info("Voice announcement system started")

    def stop(self):
        """Stop the announcement system."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        logger.info("Voice announcement system stopped")

    def _worker_loop(self):
        """Worker thread for processing announcements."""
        while self._running:
            try:
                # Get next announcement (with timeout to check running flag)
                try:
                    priority, timestamp, announcement = self._queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # Skip if expired
                if announcement.is_expired():
                    logger.debug(f"Skipping expired announcement: {announcement.id}")
                    continue

                # Check quiet hours (except critical)
                if self._is_quiet_hours() and announcement.priority != AnnouncementPriority.CRITICAL:
                    logger.debug("Skipping announcement during quiet hours")
                    continue

                # Play announcement
                self._current_announcement = announcement
                self._play_announcement(announcement)
                self._current_announcement = None

            except Exception as e:
                logger.error(f"Announcement worker error: {e}")

    def _play_announcement(self, announcement: Announcement):
        """Play a single announcement."""
        for i in range(announcement.repeat_count):
            if not self._running:
                break

            logger.info(f"Speaking: {announcement.text}")
            self.tts.speak(announcement.text)

            if i < announcement.repeat_count - 1:
                time.sleep(announcement.repeat_delay)

        self._last_announcement_time[announcement.announcement_type.value] = time.time()

    def _is_quiet_hours(self) -> bool:
        """Check if currently in quiet hours."""
        if not self.quiet_hours_enabled:
            return False

        from datetime import datetime
        current_hour = datetime.now().hour

        if self.quiet_hours_start <= self.quiet_hours_end:
            return self.quiet_hours_start <= current_hour < self.quiet_hours_end
        else:
            # Overnight quiet hours
            return current_hour >= self.quiet_hours_start or current_hour < self.quiet_hours_end

    def _should_rate_limit(self, announcement_type: AnnouncementType) -> bool:
        """Check if announcement should be rate limited."""
        last_time = self._last_announcement_time.get(announcement_type.value, 0)
        return time.time() - last_time < self._min_interval

    def announce(
        self,
        text: str,
        priority: AnnouncementPriority = AnnouncementPriority.NORMAL,
        announcement_type: AnnouncementType = AnnouncementType.ALERT,
        repeat: int = 1,
        expires_in: Optional[float] = None,
    ) -> str:
        """
        Queue an announcement.

        Args:
            text: The text to speak
            priority: Announcement priority
            announcement_type: Type of announcement
            repeat: Number of times to repeat
            expires_in: Seconds until announcement expires

        Returns:
            Announcement ID
        """
        import uuid
        announcement_id = f"ann-{uuid.uuid4().hex[:8]}"

        announcement = Announcement(
            id=announcement_id,
            text=text,
            priority=priority,
            announcement_type=announcement_type,
            repeat_count=repeat,
            expires_at=time.time() + expires_in if expires_in else None,
        )

        # Check rate limiting (skip for high priority)
        if priority.value < AnnouncementPriority.HIGH.value:
            if self._should_rate_limit(announcement_type):
                logger.debug(f"Rate limiting announcement: {announcement_id}")
                return announcement_id

        # Add to priority queue (negative priority for max-heap behavior)
        self._queue.put((-priority.value, announcement.timestamp, announcement))
        logger.debug(f"Queued announcement: {announcement_id}")

        return announcement_id

    def announce_from_template(
        self,
        template_name: str,
        priority: AnnouncementPriority = AnnouncementPriority.NORMAL,
        **kwargs,
    ) -> Optional[str]:
        """Announce using a template."""
        template = self.templates.get(template_name)
        if not template:
            logger.warning(f"Unknown template: {template_name}")
            return None

        try:
            text = template.format(**kwargs)
            return self.announce(text, priority)
        except KeyError as e:
            logger.error(f"Missing template variable: {e}")
            return None

    # Convenience methods for common announcements

    def announce_predator(
        self,
        predator_type: str,
        location: str,
        confidence: float,
    ) -> str:
        """Announce predator detection."""
        if confidence >= 0.85:
            return self.announce_from_template(
                "predator_detected_high",
                priority=AnnouncementPriority.HIGH,
                predator_type=predator_type,
                location=location,
            )
        else:
            return self.announce_from_template(
                "predator_detected",
                priority=AnnouncementPriority.NORMAL,
                predator_type=predator_type,
                location=location,
            )

    def announce_deterrence(self, method: str) -> str:
        """Announce deterrence activation."""
        return self.announce_from_template(
            "deterrence_activated",
            priority=AnnouncementPriority.NORMAL,
            method=method,
        )

    def announce_device_status(self, device_name: str, online: bool) -> str:
        """Announce device status change."""
        template = "device_online" if online else "device_offline"
        priority = AnnouncementPriority.LOW if online else AnnouncementPriority.NORMAL
        return self.announce_from_template(
            template,
            priority=priority,
            device_name=device_name,
        )

    def announce_system_status(self, armed: bool) -> str:
        """Announce system arm/disarm."""
        template = "system_armed" if armed else "system_disarmed"
        return self.announce_from_template(
            template,
            priority=AnnouncementPriority.NORMAL,
        )

    def clear_queue(self):
        """Clear all pending announcements."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        logger.info("Cleared announcement queue")

    def get_queue_size(self) -> int:
        """Get number of pending announcements."""
        return self._queue.qsize()

    def set_voice(self, voice_id: str):
        """Change the TTS voice."""
        self.config.voice_id = voice_id
        self.tts = TTSEngine(self.config)
        self.tts.initialize()

    def set_volume(self, volume: float):
        """Set announcement volume (0.0 to 1.0)."""
        self.config.volume = max(0.0, min(1.0, volume))
        self.tts = TTSEngine(self.config)
        self.tts.initialize()

    def set_rate(self, rate: int):
        """Set speech rate (words per minute)."""
        self.config.rate = max(50, min(300, rate))
        self.tts = TTSEngine(self.config)
        self.tts.initialize()
