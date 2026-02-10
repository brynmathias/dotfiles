"""
Replay engine for historical event playback.

Provides time-synchronized playback of recorded events.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Awaitable
from enum import Enum
import asyncio
import time
import logging
from pathlib import Path

from .event_store import EventStore, StoredEvent, EventType

logger = logging.getLogger(__name__)


class PlaybackSpeed(Enum):
    """Available playback speeds."""
    QUARTER = 0.25
    HALF = 0.5
    NORMAL = 1.0
    DOUBLE = 2.0
    QUAD = 4.0
    EIGHT = 8.0
    SIXTEEN = 16.0


class ReplayState(Enum):
    """Replay session state."""
    IDLE = "idle"
    PLAYING = "playing"
    PAUSED = "paused"
    SEEKING = "seeking"
    ENDED = "ended"


@dataclass
class TimelineEvent:
    """An event on the replay timeline."""
    timestamp: float
    event_type: EventType
    device_id: Optional[str]
    summary: str
    severity: str = "normal"
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplaySession:
    """A replay session with state and configuration."""
    session_id: str
    start_time: float
    end_time: float
    current_time: float
    state: ReplayState = ReplayState.IDLE
    speed: PlaybackSpeed = PlaybackSpeed.NORMAL
    event_types: List[EventType] = field(default_factory=list)
    device_ids: List[str] = field(default_factory=list)
    loop: bool = False

    def duration(self) -> float:
        """Total duration in seconds."""
        return self.end_time - self.start_time

    def progress(self) -> float:
        """Current progress as 0-1 fraction."""
        duration = self.duration()
        if duration <= 0:
            return 0
        return (self.current_time - self.start_time) / duration

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "current_time": self.current_time,
            "state": self.state.value,
            "speed": self.speed.value,
            "progress": self.progress(),
            "duration": self.duration(),
            "loop": self.loop,
        }


# Type for event callback
EventCallback = Callable[[StoredEvent], Awaitable[None]]


class ReplayEngine:
    """
    Engine for replaying historical events.

    Features:
    - Time-synchronized event playback
    - Variable playback speed
    - Pause/resume/seek functionality
    - Event filtering
    - Timeline visualization data
    """

    def __init__(self, event_store: EventStore):
        self.event_store = event_store
        self.sessions: Dict[str, ReplaySession] = {}
        self._playback_tasks: Dict[str, asyncio.Task] = {}
        self._callbacks: Dict[str, List[EventCallback]] = {}
        self._next_session_id = 1

    def create_session(
        self,
        start_time: float,
        end_time: float,
        event_types: Optional[List[EventType]] = None,
        device_ids: Optional[List[str]] = None,
    ) -> ReplaySession:
        """
        Create a new replay session.

        Args:
            start_time: Start timestamp for replay
            end_time: End timestamp for replay
            event_types: Optional filter for event types
            device_ids: Optional filter for device IDs

        Returns:
            New ReplaySession
        """
        session_id = f"replay-{self._next_session_id}"
        self._next_session_id += 1

        session = ReplaySession(
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            current_time=start_time,
            event_types=event_types or list(EventType),
            device_ids=device_ids or [],
        )

        self.sessions[session_id] = session
        self._callbacks[session_id] = []

        logger.info(f"Created replay session {session_id}: {start_time} -> {end_time}")
        return session

    def get_session(self, session_id: str) -> Optional[ReplaySession]:
        """Get a replay session by ID."""
        return self.sessions.get(session_id)

    def delete_session(self, session_id: str):
        """Delete a replay session."""
        if session_id in self._playback_tasks:
            self._playback_tasks[session_id].cancel()
            del self._playback_tasks[session_id]

        if session_id in self.sessions:
            del self.sessions[session_id]

        if session_id in self._callbacks:
            del self._callbacks[session_id]

    def add_callback(self, session_id: str, callback: EventCallback):
        """Add an event callback for a session."""
        if session_id in self._callbacks:
            self._callbacks[session_id].append(callback)

    async def play(self, session_id: str):
        """Start or resume playback."""
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        if session.state == ReplayState.PLAYING:
            return  # Already playing

        session.state = ReplayState.PLAYING

        # Cancel existing task if any
        if session_id in self._playback_tasks:
            self._playback_tasks[session_id].cancel()

        # Start playback task
        self._playback_tasks[session_id] = asyncio.create_task(
            self._playback_loop(session_id)
        )

        logger.info(f"Started playback for session {session_id}")

    async def pause(self, session_id: str):
        """Pause playback."""
        session = self.sessions.get(session_id)
        if not session:
            return

        session.state = ReplayState.PAUSED

        # Cancel playback task
        if session_id in self._playback_tasks:
            self._playback_tasks[session_id].cancel()
            del self._playback_tasks[session_id]

        logger.info(f"Paused playback for session {session_id}")

    async def seek(self, session_id: str, timestamp: float):
        """
        Seek to a specific timestamp.

        Args:
            session_id: Session ID
            timestamp: Target timestamp to seek to
        """
        session = self.sessions.get(session_id)
        if not session:
            return

        was_playing = session.state == ReplayState.PLAYING

        # Pause playback during seek
        await self.pause(session_id)

        session.state = ReplayState.SEEKING

        # Clamp timestamp to valid range
        timestamp = max(session.start_time, min(session.end_time, timestamp))
        session.current_time = timestamp

        session.state = ReplayState.PAUSED

        # Resume if was playing
        if was_playing:
            await self.play(session_id)

        logger.info(f"Seeked session {session_id} to {timestamp}")

    def set_speed(self, session_id: str, speed: PlaybackSpeed):
        """Set playback speed."""
        session = self.sessions.get(session_id)
        if session:
            session.speed = speed
            logger.info(f"Set speed for session {session_id} to {speed.value}x")

    async def _playback_loop(self, session_id: str):
        """Main playback loop."""
        session = self.sessions.get(session_id)
        if not session:
            return

        try:
            # Load events for the session
            events = self.event_store.get_events_in_range(
                start_time=session.current_time,
                end_time=session.end_time,
                event_types=session.event_types if session.event_types else None,
                device_ids=session.device_ids if session.device_ids else None,
            )

            event_index = 0
            last_real_time = time.time()
            last_replay_time = session.current_time

            while session.state == ReplayState.PLAYING:
                # Calculate elapsed time
                current_real_time = time.time()
                real_elapsed = current_real_time - last_real_time
                last_real_time = current_real_time

                # Calculate replay time advancement
                replay_elapsed = real_elapsed * session.speed.value
                session.current_time = last_replay_time + replay_elapsed
                last_replay_time = session.current_time

                # Check if ended
                if session.current_time >= session.end_time:
                    if session.loop:
                        session.current_time = session.start_time
                        last_replay_time = session.current_time
                        event_index = 0
                        events = self.event_store.get_events_in_range(
                            start_time=session.start_time,
                            end_time=session.end_time,
                            event_types=session.event_types if session.event_types else None,
                            device_ids=session.device_ids if session.device_ids else None,
                        )
                    else:
                        session.state = ReplayState.ENDED
                        break

                # Emit events up to current time
                while event_index < len(events):
                    event = events[event_index]
                    if event.timestamp <= session.current_time:
                        await self._emit_event(session_id, event)
                        event_index += 1
                    else:
                        break

                # Small sleep to prevent busy loop
                await asyncio.sleep(0.01)  # 100 Hz update rate

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Playback error for session {session_id}: {e}")
            session.state = ReplayState.PAUSED

    async def _emit_event(self, session_id: str, event: StoredEvent):
        """Emit an event to all registered callbacks."""
        callbacks = self._callbacks.get(session_id, [])
        for callback in callbacks:
            try:
                await callback(event)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def get_timeline(
        self,
        session_id: str,
        bucket_seconds: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Get timeline data for visualization.

        Returns buckets of events for the session's time range.
        """
        session = self.sessions.get(session_id)
        if not session:
            return []

        return self.event_store.get_timeline_summary(
            start_time=session.start_time,
            end_time=session.end_time,
            bucket_seconds=bucket_seconds,
        )

    def get_events_at_time(
        self,
        session_id: str,
        timestamp: float,
        window_seconds: float = 5.0,
    ) -> List[TimelineEvent]:
        """
        Get events around a specific timestamp.

        Args:
            session_id: Session ID
            timestamp: Center timestamp
            window_seconds: Time window around timestamp

        Returns:
            List of timeline events
        """
        session = self.sessions.get(session_id)
        if not session:
            return []

        events = self.event_store.get_events_in_range(
            start_time=timestamp - window_seconds / 2,
            end_time=timestamp + window_seconds / 2,
            event_types=session.event_types if session.event_types else None,
            device_ids=session.device_ids if session.device_ids else None,
        )

        return [self._to_timeline_event(e) for e in events]

    def _to_timeline_event(self, event: StoredEvent) -> TimelineEvent:
        """Convert StoredEvent to TimelineEvent for display."""
        summary = self._generate_event_summary(event)
        severity = self._determine_severity(event)

        return TimelineEvent(
            timestamp=event.timestamp,
            event_type=event.event_type,
            device_id=event.device_id,
            summary=summary,
            severity=severity,
            data=event.data,
        )

    def _generate_event_summary(self, event: StoredEvent) -> str:
        """Generate a human-readable summary for an event."""
        if event.event_type == EventType.DETECTION:
            predator = event.data.get("predator_type", "unknown")
            confidence = event.data.get("confidence", 0) * 100
            return f"Detected {predator} ({confidence:.0f}% confidence)"

        elif event.event_type == EventType.ALERT:
            severity = event.data.get("severity", "unknown")
            predator = event.data.get("predator_type", "threat")
            return f"{severity.capitalize()} alert: {predator}"

        elif event.event_type == EventType.DETERRENCE:
            action = event.data.get("action_type", "unknown")
            success = "successful" if event.data.get("success") else "attempted"
            return f"Deterrence {action} - {success}"

        elif event.event_type == EventType.DEVICE_STATUS:
            status = event.data.get("status", "unknown")
            return f"Device {event.device_id}: {status}"

        elif event.event_type == EventType.TRACK_UPDATE:
            predator = event.data.get("predator_type", "unknown")
            return f"Tracking {predator}"

        else:
            return f"{event.event_type.value} event"

    def _determine_severity(self, event: StoredEvent) -> str:
        """Determine display severity for an event."""
        if event.event_type == EventType.ALERT:
            return event.data.get("severity", "medium")

        if event.event_type == EventType.DETECTION:
            confidence = event.data.get("confidence", 0)
            if confidence >= 0.9:
                return "high"
            elif confidence >= 0.7:
                return "medium"
            return "low"

        if event.event_type == EventType.DEVICE_STATUS:
            status = event.data.get("status", "")
            if status == "offline":
                return "high"
            elif status == "warning":
                return "medium"
            return "normal"

        return "normal"


class ReplayAPI:
    """
    API helper for replay functionality.

    Provides methods for REST API integration.
    """

    def __init__(self, replay_engine: ReplayEngine):
        self.engine = replay_engine

    async def create_session(
        self,
        start_time: float,
        end_time: float,
        event_types: Optional[List[str]] = None,
        device_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a new replay session via API."""
        types = None
        if event_types:
            types = [EventType(t) for t in event_types]

        session = self.engine.create_session(
            start_time=start_time,
            end_time=end_time,
            event_types=types,
            device_ids=device_ids,
        )

        return session.to_dict()

    async def control(
        self,
        session_id: str,
        action: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Control replay playback.

        Actions: play, pause, seek, speed
        """
        session = self.engine.get_session(session_id)
        if not session:
            return {"error": "Session not found"}

        if action == "play":
            await self.engine.play(session_id)
        elif action == "pause":
            await self.engine.pause(session_id)
        elif action == "seek":
            timestamp = kwargs.get("timestamp")
            if timestamp is not None:
                await self.engine.seek(session_id, float(timestamp))
        elif action == "speed":
            speed_value = kwargs.get("speed", 1.0)
            speed = PlaybackSpeed(float(speed_value))
            self.engine.set_speed(session_id, speed)
        else:
            return {"error": f"Unknown action: {action}"}

        # Return updated session state
        session = self.engine.get_session(session_id)
        return session.to_dict() if session else {"error": "Session not found"}

    async def get_timeline(
        self,
        session_id: str,
        bucket_seconds: int = 60,
    ) -> Dict[str, Any]:
        """Get timeline data for a session."""
        session = self.engine.get_session(session_id)
        if not session:
            return {"error": "Session not found"}

        timeline = self.engine.get_timeline(session_id, bucket_seconds)

        return {
            "session": session.to_dict(),
            "timeline": timeline,
        }

    async def get_events(
        self,
        session_id: str,
        timestamp: Optional[float] = None,
        window_seconds: float = 5.0,
    ) -> Dict[str, Any]:
        """Get events around a timestamp."""
        session = self.engine.get_session(session_id)
        if not session:
            return {"error": "Session not found"}

        if timestamp is None:
            timestamp = session.current_time

        events = self.engine.get_events_at_time(
            session_id=session_id,
            timestamp=timestamp,
            window_seconds=window_seconds,
        )

        return {
            "timestamp": timestamp,
            "events": [
                {
                    "timestamp": e.timestamp,
                    "type": e.event_type.value,
                    "device_id": e.device_id,
                    "summary": e.summary,
                    "severity": e.severity,
                    "data": e.data,
                }
                for e in events
            ],
        }
