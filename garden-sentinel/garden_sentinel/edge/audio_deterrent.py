"""
Audio deterrent system for Garden Sentinel.

Provides various audio-based deterrents:
- Predator distress calls
- Dog barking sounds
- Ultrasonic deterrents
- Custom sound playback
- Text-to-speech announcements
"""

import asyncio
import logging
import os
import random
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Callable
import wave
import struct
import math

logger = logging.getLogger(__name__)


class SoundType(Enum):
    """Types of deterrent sounds."""
    DOG_BARK = "dog_bark"
    FOX_DISTRESS = "fox_distress"
    PREDATOR_GROWL = "predator_growl"
    ULTRASONIC = "ultrasonic"
    ALARM = "alarm"
    CUSTOM = "custom"
    TTS = "tts"  # Text-to-speech


@dataclass
class AudioConfig:
    """Audio deterrent configuration."""
    enabled: bool = True
    sounds_dir: str = "/opt/garden-sentinel/sounds"
    default_volume: float = 0.8  # 0.0 - 1.0
    max_volume: float = 1.0

    # Cooldown between plays
    cooldown_seconds: float = 5.0

    # Sound durations
    default_duration: float = 3.0
    max_duration: float = 30.0

    # Ultrasonic settings
    ultrasonic_frequency: int = 20000  # Hz
    ultrasonic_duration: float = 2.0

    # TTS settings
    tts_engine: str = "espeak"  # espeak, pico2wave, festival
    tts_voice: str = "en"
    tts_speed: int = 150

    # Hardware
    audio_device: str = "default"  # ALSA device

    # Effectiveness tracking
    track_effectiveness: bool = True


@dataclass
class SoundFile:
    """Metadata for a sound file."""
    sound_id: str
    sound_type: SoundType
    file_path: str
    duration: float
    description: str = ""
    effectiveness_score: float = 0.5  # 0.0 - 1.0, learned over time

    # Usage statistics
    play_count: int = 0
    last_played: Optional[float] = None


class AudioPlayer:
    """
    Low-level audio playback using ALSA or PulseAudio.

    Handles actual sound output to speakers.
    """

    def __init__(self, device: str = "default"):
        self.device = device
        self._current_process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def play_file(
        self,
        file_path: str,
        volume: float = 1.0,
        blocking: bool = False,
    ) -> bool:
        """
        Play an audio file.

        Supports: .wav, .mp3, .ogg
        """
        if not os.path.exists(file_path):
            logger.error(f"Audio file not found: {file_path}")
            return False

        with self._lock:
            # Stop any currently playing sound
            self.stop()

            try:
                # Determine player based on file type
                ext = Path(file_path).suffix.lower()

                if ext == ".wav":
                    cmd = ["aplay", "-D", self.device, "-q", file_path]
                elif ext in (".mp3", ".ogg"):
                    # Use mpv or ffplay for other formats
                    vol_percent = int(volume * 100)
                    cmd = ["mpv", "--no-video", f"--volume={vol_percent}",
                           f"--audio-device=alsa/{self.device}", file_path]
                else:
                    logger.error(f"Unsupported audio format: {ext}")
                    return False

                if blocking:
                    subprocess.run(cmd, capture_output=True)
                else:
                    self._current_process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )

                return True

            except FileNotFoundError as e:
                logger.error(f"Audio player not found: {e}")
                return False
            except Exception as e:
                logger.error(f"Error playing audio: {e}")
                return False

    def play_tone(
        self,
        frequency: int,
        duration: float,
        volume: float = 1.0,
    ) -> bool:
        """
        Generate and play a pure tone.

        Useful for ultrasonic deterrents (if speakers support it).
        """
        try:
            # Generate tone as WAV in memory
            sample_rate = 44100
            num_samples = int(sample_rate * duration)

            # Generate sine wave
            samples = []
            for i in range(num_samples):
                t = i / sample_rate
                value = volume * math.sin(2 * math.pi * frequency * t)
                # Convert to 16-bit signed integer
                samples.append(int(value * 32767))

            # Write to temporary WAV file
            temp_file = "/tmp/garden_sentinel_tone.wav"
            with wave.open(temp_file, 'w') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(struct.pack(f'{len(samples)}h', *samples))

            return self.play_file(temp_file, volume=1.0, blocking=True)

        except Exception as e:
            logger.error(f"Error generating tone: {e}")
            return False

    def stop(self):
        """Stop currently playing sound."""
        if self._current_process:
            self._current_process.terminate()
            try:
                self._current_process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._current_process.kill()
            self._current_process = None

    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        if self._current_process:
            return self._current_process.poll() is None
        return False


class TTSEngine:
    """
    Text-to-speech engine for voice announcements.
    """

    def __init__(
        self,
        engine: str = "espeak",
        voice: str = "en",
        speed: int = 150,
    ):
        self.engine = engine
        self.voice = voice
        self.speed = speed

    def speak(self, text: str, blocking: bool = True) -> bool:
        """
        Convert text to speech and play it.
        """
        try:
            if self.engine == "espeak":
                cmd = [
                    "espeak",
                    "-v", self.voice,
                    "-s", str(self.speed),
                    text,
                ]
            elif self.engine == "pico2wave":
                temp_file = "/tmp/garden_sentinel_tts.wav"
                # Generate WAV
                subprocess.run([
                    "pico2wave",
                    "-l", self.voice,
                    "-w", temp_file,
                    text,
                ], check=True)
                cmd = ["aplay", "-q", temp_file]
            elif self.engine == "festival":
                cmd = ["festival", "--tts"]
                # Festival reads from stdin
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                proc.communicate(input=text.encode())
                return proc.returncode == 0
            else:
                logger.error(f"Unknown TTS engine: {self.engine}")
                return False

            if blocking:
                result = subprocess.run(cmd, capture_output=True)
                return result.returncode == 0
            else:
                subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True

        except FileNotFoundError:
            logger.error(f"TTS engine not found: {self.engine}")
            return False
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return False

    def generate_file(self, text: str, output_path: str) -> bool:
        """Generate speech to a file."""
        try:
            if self.engine == "espeak":
                cmd = [
                    "espeak",
                    "-v", self.voice,
                    "-s", str(self.speed),
                    "-w", output_path,
                    text,
                ]
            elif self.engine == "pico2wave":
                cmd = [
                    "pico2wave",
                    "-l", self.voice,
                    "-w", output_path,
                    text,
                ]
            else:
                return False

            result = subprocess.run(cmd, capture_output=True)
            return result.returncode == 0

        except Exception as e:
            logger.error(f"Error generating TTS file: {e}")
            return False


class AudioDeterrent:
    """
    Main audio deterrent controller.

    Manages sound playback, effectiveness tracking, and smart sound selection.
    """

    def __init__(self, config: AudioConfig):
        self.config = config
        self.player = AudioPlayer(device=config.audio_device)
        self.tts = TTSEngine(
            engine=config.tts_engine,
            voice=config.tts_voice,
            speed=config.tts_speed,
        )

        # Sound library
        self.sounds: dict[str, SoundFile] = {}
        self._load_sounds()

        # Cooldown tracking
        self._last_play_time: dict[str, float] = {}

        # Effectiveness tracking
        self._effectiveness_history: list[dict] = []

        # Callbacks
        self._on_play_callbacks: list[Callable] = []

    def _load_sounds(self):
        """Load available sounds from the sounds directory."""
        sounds_dir = Path(self.config.sounds_dir)
        if not sounds_dir.exists():
            logger.warning(f"Sounds directory not found: {sounds_dir}")
            sounds_dir.mkdir(parents=True, exist_ok=True)
            return

        # Scan for sound files
        for sound_type in SoundType:
            type_dir = sounds_dir / sound_type.value
            if type_dir.exists():
                for file_path in type_dir.glob("*"):
                    if file_path.suffix.lower() in (".wav", ".mp3", ".ogg"):
                        sound_id = f"{sound_type.value}_{file_path.stem}"
                        self.sounds[sound_id] = SoundFile(
                            sound_id=sound_id,
                            sound_type=sound_type,
                            file_path=str(file_path),
                            duration=self._get_duration(file_path),
                            description=file_path.stem.replace("_", " ").title(),
                        )

        logger.info(f"Loaded {len(self.sounds)} deterrent sounds")

    def _get_duration(self, file_path: Path) -> float:
        """Get duration of audio file."""
        try:
            if file_path.suffix.lower() == ".wav":
                with wave.open(str(file_path), 'r') as wav:
                    return wav.getnframes() / wav.getframerate()
            else:
                # Use ffprobe for other formats
                result = subprocess.run([
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(file_path),
                ], capture_output=True, text=True)
                return float(result.stdout.strip())
        except Exception:
            return 3.0  # Default

    def add_play_callback(self, callback: Callable):
        """Add callback for when a sound is played."""
        self._on_play_callbacks.append(callback)

    def _check_cooldown(self, sound_id: str) -> bool:
        """Check if cooldown has elapsed for a sound."""
        last_time = self._last_play_time.get(sound_id, 0)
        return time.time() - last_time >= self.config.cooldown_seconds

    def play(
        self,
        sound_type: SoundType,
        sound_id: Optional[str] = None,
        volume: Optional[float] = None,
        predator_type: Optional[str] = None,
    ) -> bool:
        """
        Play a deterrent sound.

        Args:
            sound_type: Type of sound to play
            sound_id: Specific sound ID (if None, selects best one)
            volume: Volume level (0.0 - 1.0)
            predator_type: Type of predator (for smart selection)
        """
        if not self.config.enabled:
            return False

        volume = volume or self.config.default_volume
        volume = min(volume, self.config.max_volume)

        # Select sound
        if sound_id and sound_id in self.sounds:
            sound = self.sounds[sound_id]
        else:
            sound = self._select_sound(sound_type, predator_type)

        if not sound:
            logger.warning(f"No sounds available for type: {sound_type}")
            return False

        # Check cooldown
        if not self._check_cooldown(sound.sound_id):
            logger.debug(f"Sound on cooldown: {sound.sound_id}")
            return False

        # Play sound
        logger.info(f"Playing deterrent: {sound.sound_id} (volume: {volume:.0%})")
        success = self.player.play_file(sound.file_path, volume=volume)

        if success:
            # Update tracking
            sound.play_count += 1
            sound.last_played = time.time()
            self._last_play_time[sound.sound_id] = time.time()

            # Trigger callbacks
            for callback in self._on_play_callbacks:
                try:
                    callback(sound, predator_type)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

        return success

    def _select_sound(
        self,
        sound_type: SoundType,
        predator_type: Optional[str] = None,
    ) -> Optional[SoundFile]:
        """
        Select the best sound to play.

        Uses effectiveness scores and randomization.
        """
        # Get sounds of this type
        candidates = [
            s for s in self.sounds.values()
            if s.sound_type == sound_type
        ]

        if not candidates:
            return None

        # Weight by effectiveness score
        if self.config.track_effectiveness:
            # Weighted random selection
            total_weight = sum(s.effectiveness_score + 0.1 for s in candidates)
            r = random.uniform(0, total_weight)

            cumulative = 0
            for sound in candidates:
                cumulative += sound.effectiveness_score + 0.1
                if r <= cumulative:
                    return sound

        # Fallback to random
        return random.choice(candidates)

    def play_ultrasonic(
        self,
        frequency: Optional[int] = None,
        duration: Optional[float] = None,
    ) -> bool:
        """
        Play ultrasonic deterrent.

        Note: Effectiveness depends on speaker capability.
        Most speakers can't produce true ultrasonic frequencies.
        """
        freq = frequency or self.config.ultrasonic_frequency
        dur = duration or self.config.ultrasonic_duration

        # Clamp to reasonable range
        freq = min(freq, 22000)  # Most speakers max out around 20kHz

        logger.info(f"Playing ultrasonic: {freq}Hz for {dur}s")
        return self.player.play_tone(freq, dur, volume=self.config.max_volume)

    def play_alarm(self, duration: float = 5.0) -> bool:
        """Play alarm sound."""
        return self.play(SoundType.ALARM)

    def announce(self, text: str, blocking: bool = False) -> bool:
        """
        Make a voice announcement.

        Example: "Warning! Intruder detected in the garden."
        """
        if not self.config.enabled:
            return False

        logger.info(f"TTS announcement: {text}")
        return self.tts.speak(text, blocking=blocking)

    def play_for_predator(
        self,
        predator_type: str,
        threat_level: str = "high",
    ) -> bool:
        """
        Play appropriate deterrent for a specific predator type.

        Uses predator-specific sounds when available.
        """
        # Map predator types to sound types
        predator_sounds = {
            "fox": [SoundType.FOX_DISTRESS, SoundType.DOG_BARK],
            "badger": [SoundType.DOG_BARK, SoundType.PREDATOR_GROWL],
            "cat": [SoundType.DOG_BARK, SoundType.ULTRASONIC],
            "bird_of_prey": [SoundType.ALARM],
            "rat": [SoundType.ULTRASONIC, SoundType.PREDATOR_GROWL],
            "mink": [SoundType.DOG_BARK, SoundType.FOX_DISTRESS],
        }

        sound_types = predator_sounds.get(predator_type, [SoundType.DOG_BARK])

        # Adjust volume based on threat level
        volume = self.config.default_volume
        if threat_level == "critical":
            volume = self.config.max_volume
        elif threat_level == "high":
            volume = min(self.config.default_volume + 0.2, self.config.max_volume)

        # Try each sound type until one works
        for sound_type in sound_types:
            if self.play(sound_type, volume=volume, predator_type=predator_type):
                return True

        return False

    def report_effectiveness(
        self,
        sound_id: str,
        was_effective: bool,
        predator_type: Optional[str] = None,
    ):
        """
        Report whether a deterrent was effective.

        Called when we know if the predator left after the sound.
        """
        if not self.config.track_effectiveness:
            return

        if sound_id not in self.sounds:
            return

        sound = self.sounds[sound_id]

        # Update effectiveness score using exponential moving average
        alpha = 0.2  # Learning rate
        new_value = 1.0 if was_effective else 0.0
        sound.effectiveness_score = (
            alpha * new_value + (1 - alpha) * sound.effectiveness_score
        )

        # Record in history
        self._effectiveness_history.append({
            "sound_id": sound_id,
            "predator_type": predator_type,
            "was_effective": was_effective,
            "timestamp": time.time(),
        })

        # Keep history bounded
        if len(self._effectiveness_history) > 1000:
            self._effectiveness_history = self._effectiveness_history[-500:]

        logger.debug(
            f"Updated effectiveness for {sound_id}: "
            f"{sound.effectiveness_score:.2f}"
        )

    def get_effectiveness_report(self) -> dict:
        """Get effectiveness statistics for all sounds."""
        report = {}
        for sound_id, sound in self.sounds.items():
            report[sound_id] = {
                "type": sound.sound_type.value,
                "description": sound.description,
                "play_count": sound.play_count,
                "effectiveness_score": sound.effectiveness_score,
            }
        return report

    def stop(self):
        """Stop any currently playing sound."""
        self.player.stop()

    def test_audio(self) -> bool:
        """Test that audio output is working."""
        return self.tts.speak("Audio test successful", blocking=True)
