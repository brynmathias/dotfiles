"""
Zone entry/exit tracking for predator movement analysis.

Tracks predator positions over time and detects zone transitions.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from collections import defaultdict

from .garden_map import GardenMap, Point, Zone


class ZoneEventType(Enum):
    """Type of zone transition event."""
    ENTERED = "entered"
    EXITED = "exited"
    PASSED_THROUGH = "passed_through"  # Quick entry and exit


@dataclass
class ZoneEvent:
    """Records a zone entry or exit event."""
    track_id: str
    zone_id: str
    zone_name: str
    event_type: ZoneEventType
    timestamp: float
    position: Point
    predator_type: str

    # For exit events, how long they were in the zone
    duration_in_zone: Optional[float] = None

    # Entry point if this is an entry event to a protected zone
    entry_direction: Optional[str] = None  # "north", "south", "east", "west"


@dataclass
class PositionUpdate:
    """A single position update for a tracked predator."""
    position: Point
    timestamp: float
    confidence: float = 1.0


@dataclass
class PredatorTrack:
    """Tracks a single predator's movement through the garden."""
    track_id: str
    predator_type: str
    first_seen: float
    last_seen: float
    positions: list[PositionUpdate] = field(default_factory=list)

    # Zone state tracking
    current_zones: set[str] = field(default_factory=set)  # Zone IDs currently in
    zone_entry_times: dict[str, float] = field(default_factory=dict)  # zone_id -> entry time

    # Movement analysis
    total_distance: float = 0.0
    was_sprayed: bool = False
    spray_time: Optional[float] = None

    def add_position(self, position: Point, timestamp: float, confidence: float = 1.0):
        """Add a new position to the track."""
        if self.positions:
            # Calculate distance from last position
            last_pos = self.positions[-1].position
            self.total_distance += position.distance_to(last_pos)

        self.positions.append(PositionUpdate(position, timestamp, confidence))
        self.last_seen = timestamp

    @property
    def current_position(self) -> Optional[Point]:
        """Get the most recent position."""
        return self.positions[-1].position if self.positions else None

    @property
    def velocity(self) -> Optional[tuple[float, float]]:
        """Estimate current velocity (m/s) from recent positions."""
        if len(self.positions) < 2:
            return None

        # Use last few positions for smoothing
        recent = self.positions[-min(5, len(self.positions)):]
        if recent[-1].timestamp == recent[0].timestamp:
            return (0.0, 0.0)

        dt = recent[-1].timestamp - recent[0].timestamp
        dx = recent[-1].position.x - recent[0].position.x
        dy = recent[-1].position.y - recent[0].position.y

        return (dx / dt, dy / dt)

    def get_trajectory(self) -> list[Point]:
        """Get the full trajectory as a list of points."""
        return [p.position for p in self.positions]


class ZoneTracker:
    """
    Tracks predator movements and detects zone transitions.

    Monitors predator positions and generates events when they:
    - Enter a zone (especially protected zones)
    - Exit a zone
    - Pass through entry points
    """

    def __init__(
        self,
        garden_map: GardenMap,
        track_timeout: float = 60.0,  # Tracks expire after this many seconds
        min_zone_duration: float = 1.0,  # Minimum time to count as "in zone"
    ):
        self.garden_map = garden_map
        self.track_timeout = track_timeout
        self.min_zone_duration = min_zone_duration

        # Active tracks: track_id -> PredatorTrack
        self.active_tracks: dict[str, PredatorTrack] = {}

        # Historical events
        self.events: list[ZoneEvent] = []

        # Statistics
        self.zone_entry_counts: dict[str, int] = defaultdict(int)
        self.entry_point_counts: dict[str, int] = defaultdict(int)

        # Callbacks for real-time events
        self._event_callbacks: list = []

    def register_event_callback(self, callback):
        """Register a callback for zone events. Callback receives ZoneEvent."""
        self._event_callbacks.append(callback)

    def update_position(
        self,
        track_id: str,
        predator_type: str,
        position: Point,
        timestamp: Optional[float] = None,
        confidence: float = 1.0,
    ) -> list[ZoneEvent]:
        """
        Update a predator's position and check for zone transitions.

        Returns list of zone events triggered by this update.
        """
        timestamp = timestamp or time.time()
        events = []

        # Get or create track
        if track_id not in self.active_tracks:
            track = PredatorTrack(
                track_id=track_id,
                predator_type=predator_type,
                first_seen=timestamp,
                last_seen=timestamp,
            )
            self.active_tracks[track_id] = track
        else:
            track = self.active_tracks[track_id]

        # Store previous zones before update
        previous_zones = track.current_zones.copy()

        # Add position to track
        track.add_position(position, timestamp, confidence)

        # Determine which zones the predator is now in
        current_zones = set()
        for zone in self.garden_map.zones:
            if zone.polygon.contains_point(position):
                current_zones.add(zone.zone_id)

        # Detect zone entries
        entered_zones = current_zones - previous_zones
        for zone_id in entered_zones:
            zone = self._get_zone(zone_id)
            if zone:
                track.current_zones.add(zone_id)
                track.zone_entry_times[zone_id] = timestamp

                event = ZoneEvent(
                    track_id=track_id,
                    zone_id=zone_id,
                    zone_name=zone.name,
                    event_type=ZoneEventType.ENTERED,
                    timestamp=timestamp,
                    position=position,
                    predator_type=predator_type,
                    entry_direction=self._calculate_entry_direction(track, zone),
                )
                events.append(event)
                self.zone_entry_counts[zone_id] += 1

                # Check if this is an entry point
                if zone.zone_type.value == "entry_point":
                    self.entry_point_counts[zone_id] += 1

        # Detect zone exits
        exited_zones = previous_zones - current_zones
        for zone_id in exited_zones:
            zone = self._get_zone(zone_id)
            if zone:
                entry_time = track.zone_entry_times.get(zone_id, timestamp)
                duration = timestamp - entry_time

                track.current_zones.discard(zone_id)

                # Only record if they were in zone long enough
                if duration >= self.min_zone_duration:
                    event = ZoneEvent(
                        track_id=track_id,
                        zone_id=zone_id,
                        zone_name=zone.name,
                        event_type=ZoneEventType.EXITED,
                        timestamp=timestamp,
                        position=position,
                        predator_type=predator_type,
                        duration_in_zone=duration,
                    )
                    events.append(event)
                else:
                    # Quick pass-through
                    event = ZoneEvent(
                        track_id=track_id,
                        zone_id=zone_id,
                        zone_name=zone.name,
                        event_type=ZoneEventType.PASSED_THROUGH,
                        timestamp=timestamp,
                        position=position,
                        predator_type=predator_type,
                        duration_in_zone=duration,
                    )
                    events.append(event)

        # Store events and trigger callbacks
        for event in events:
            self.events.append(event)
            for callback in self._event_callbacks:
                try:
                    callback(event)
                except Exception:
                    pass  # Don't let callback errors break tracking

        return events

    def mark_sprayed(self, track_id: str, timestamp: Optional[float] = None):
        """Mark that a tracked predator was sprayed."""
        if track_id in self.active_tracks:
            track = self.active_tracks[track_id]
            track.was_sprayed = True
            track.spray_time = timestamp or time.time()

    def check_deterred(self, track_id: str, protected_zone_id: str, window_seconds: float = 30.0) -> bool:
        """
        Check if a predator was deterred (left protected zone within window after spray).

        Returns True if the predator:
        1. Was sprayed
        2. Left the protected zone within window_seconds after spray
        """
        if track_id not in self.active_tracks:
            return False

        track = self.active_tracks[track_id]
        if not track.was_sprayed or track.spray_time is None:
            return False

        # Look for exit event from protected zone after spray
        for event in reversed(self.events):
            if event.track_id != track_id:
                continue
            if event.zone_id != protected_zone_id:
                continue
            if event.event_type != ZoneEventType.EXITED:
                continue

            # Check if exit was within window after spray
            time_after_spray = event.timestamp - track.spray_time
            if 0 < time_after_spray <= window_seconds:
                return True

        return False

    def get_track(self, track_id: str) -> Optional[PredatorTrack]:
        """Get a track by ID."""
        return self.active_tracks.get(track_id)

    def get_active_tracks(self) -> list[PredatorTrack]:
        """Get all currently active tracks."""
        return list(self.active_tracks.values())

    def get_tracks_in_zone(self, zone_id: str) -> list[PredatorTrack]:
        """Get all tracks currently in a specific zone."""
        return [
            track for track in self.active_tracks.values()
            if zone_id in track.current_zones
        ]

    def cleanup_expired_tracks(self, current_time: Optional[float] = None) -> list[PredatorTrack]:
        """
        Remove tracks that haven't been updated recently.

        Returns list of expired tracks.
        """
        current_time = current_time or time.time()
        expired = []

        for track_id, track in list(self.active_tracks.items()):
            if current_time - track.last_seen > self.track_timeout:
                expired.append(track)
                del self.active_tracks[track_id]

        return expired

    def get_entry_point_statistics(self) -> dict[str, dict]:
        """
        Get statistics about which entry points are most used.

        Returns dict mapping entry point zone_id to stats.
        """
        stats = {}
        for zone in self.garden_map.zones:
            if zone.zone_type.value == "entry_point":
                stats[zone.zone_id] = {
                    "name": zone.name,
                    "entry_count": self.entry_point_counts.get(zone.zone_id, 0),
                }

        return stats

    def get_zone_statistics(self) -> dict[str, dict]:
        """
        Get entry statistics for all zones.

        Returns dict mapping zone_id to stats.
        """
        stats = {}
        for zone in self.garden_map.zones:
            # Count exits to calculate average duration
            zone_exits = [
                e for e in self.events
                if e.zone_id == zone.zone_id and e.event_type == ZoneEventType.EXITED
            ]

            durations = [e.duration_in_zone for e in zone_exits if e.duration_in_zone]
            avg_duration = sum(durations) / len(durations) if durations else 0

            stats[zone.zone_id] = {
                "name": zone.name,
                "zone_type": zone.zone_type.value,
                "entry_count": self.zone_entry_counts.get(zone.zone_id, 0),
                "current_occupants": len(self.get_tracks_in_zone(zone.zone_id)),
                "average_duration": avg_duration,
            }

        return stats

    def get_recent_events(self, limit: int = 100) -> list[ZoneEvent]:
        """Get most recent zone events."""
        return self.events[-limit:]

    def _get_zone(self, zone_id: str) -> Optional[Zone]:
        """Get a zone by ID."""
        for zone in self.garden_map.zones:
            if zone.zone_id == zone_id:
                return zone
        return None

    def _calculate_entry_direction(self, track: PredatorTrack, zone: Zone) -> Optional[str]:
        """Determine which direction the predator entered from."""
        if len(track.positions) < 2:
            return None

        # Get recent movement vector
        recent = track.positions[-min(3, len(track.positions)):]
        dx = recent[-1].position.x - recent[0].position.x
        dy = recent[-1].position.y - recent[0].position.y

        # Determine primary direction (where they came FROM)
        if abs(dx) > abs(dy):
            return "west" if dx > 0 else "east"
        else:
            return "south" if dy > 0 else "north"


@dataclass
class DeterrenceResult:
    """Result of checking if a predator was deterred."""
    track_id: str
    predator_type: str
    was_sprayed: bool
    left_protected_zone: bool
    time_to_leave: Optional[float]  # Seconds after spray
    is_deterred: bool  # True if left within window


class DeterrenceTracker:
    """
    Specialized tracker for measuring spray deterrence effectiveness.

    Integrates with ZoneTracker to determine when predators leave
    protected zones after being sprayed.
    """

    def __init__(
        self,
        zone_tracker: ZoneTracker,
        protected_zone_ids: list[str],
        deterrence_window: float = 30.0,
    ):
        self.zone_tracker = zone_tracker
        self.protected_zone_ids = set(protected_zone_ids)
        self.deterrence_window = deterrence_window

        # Results tracking
        self.spray_events: list[dict] = []
        self.deterrence_results: list[DeterrenceResult] = []

        # Register callback for zone exits
        zone_tracker.register_event_callback(self._on_zone_event)

    def record_spray(self, track_id: str, timestamp: Optional[float] = None):
        """Record that a predator was sprayed."""
        timestamp = timestamp or time.time()

        self.zone_tracker.mark_sprayed(track_id, timestamp)

        track = self.zone_tracker.get_track(track_id)
        if track:
            self.spray_events.append({
                "track_id": track_id,
                "predator_type": track.predator_type,
                "timestamp": timestamp,
                "position": track.current_position,
                "zones": list(track.current_zones),
            })

    def _on_zone_event(self, event: ZoneEvent):
        """Handle zone events to check for deterrence."""
        if event.event_type != ZoneEventType.EXITED:
            return

        if event.zone_id not in self.protected_zone_ids:
            return

        track = self.zone_tracker.get_track(event.track_id)
        if not track or not track.was_sprayed:
            return

        # Calculate time from spray to exit
        time_to_leave = event.timestamp - track.spray_time if track.spray_time else None
        is_deterred = time_to_leave is not None and 0 < time_to_leave <= self.deterrence_window

        result = DeterrenceResult(
            track_id=event.track_id,
            predator_type=event.predator_type,
            was_sprayed=True,
            left_protected_zone=True,
            time_to_leave=time_to_leave,
            is_deterred=is_deterred,
        )
        self.deterrence_results.append(result)

    def get_effectiveness(self) -> dict:
        """
        Calculate overall spray effectiveness.

        Returns dict with effectiveness metrics.
        """
        if not self.spray_events:
            return {
                "total_sprays": 0,
                "deterred_count": 0,
                "effectiveness": 0.0,
            }

        deterred = [r for r in self.deterrence_results if r.is_deterred]

        return {
            "total_sprays": len(self.spray_events),
            "deterred_count": len(deterred),
            "effectiveness": len(deterred) / len(self.spray_events),
            "average_time_to_leave": (
                sum(r.time_to_leave for r in deterred if r.time_to_leave) / len(deterred)
                if deterred else None
            ),
        }

    def get_effectiveness_by_predator(self) -> dict[str, dict]:
        """Get effectiveness broken down by predator type."""
        from collections import defaultdict

        by_type: dict[str, dict] = defaultdict(lambda: {"sprays": 0, "deterred": 0})

        for event in self.spray_events:
            by_type[event["predator_type"]]["sprays"] += 1

        for result in self.deterrence_results:
            if result.is_deterred:
                by_type[result.predator_type]["deterred"] += 1

        return {
            predator_type: {
                "sprays": stats["sprays"],
                "deterred": stats["deterred"],
                "effectiveness": stats["deterred"] / stats["sprays"] if stats["sprays"] > 0 else 0.0,
            }
            for predator_type, stats in by_type.items()
        }
