"""
Patrol scheduler for automated drone patrols.

Manages scheduled and triggered patrol missions.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable
from enum import Enum
from datetime import datetime, time as dt_time, timedelta
import asyncio
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PatrolWaypoint:
    """A waypoint in a patrol route."""
    id: str
    lat: float
    lng: float
    altitude: float  # meters above ground
    heading: Optional[float] = None  # degrees, None for auto
    hover_time: float = 0.0  # seconds to hover at waypoint
    action: Optional[str] = None  # e.g., "photo", "video", "scan"
    action_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "lat": self.lat,
            "lng": self.lng,
            "altitude": self.altitude,
            "heading": self.heading,
            "hover_time": self.hover_time,
            "action": self.action,
            "action_params": self.action_params,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatrolWaypoint":
        return cls(
            id=data["id"],
            lat=data["lat"],
            lng=data["lng"],
            altitude=data["altitude"],
            heading=data.get("heading"),
            hover_time=data.get("hover_time", 0.0),
            action=data.get("action"),
            action_params=data.get("action_params", {}),
        )


@dataclass
class PatrolRoute:
    """A patrol route consisting of waypoints."""
    id: str
    name: str
    waypoints: List[PatrolWaypoint]
    return_to_home: bool = True
    max_speed: float = 5.0  # m/s
    default_altitude: float = 20.0  # meters

    def total_distance(self) -> float:
        """Calculate total route distance in meters."""
        import math

        def haversine(lat1, lon1, lat2, lon2):
            R = 6371000  # Earth radius in meters
            phi1, phi2 = math.radians(lat1), math.radians(lat2)
            dphi = math.radians(lat2 - lat1)
            dlambda = math.radians(lon2 - lon1)
            a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
            return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance = 0.0
        for i in range(1, len(self.waypoints)):
            wp1, wp2 = self.waypoints[i - 1], self.waypoints[i]
            distance += haversine(wp1.lat, wp1.lng, wp2.lat, wp2.lng)
        return distance

    def estimated_duration(self) -> float:
        """Estimate mission duration in seconds."""
        distance = self.total_distance()
        travel_time = distance / self.max_speed
        hover_time = sum(wp.hover_time for wp in self.waypoints)
        return travel_time + hover_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "waypoints": [wp.to_dict() for wp in self.waypoints],
            "return_to_home": self.return_to_home,
            "max_speed": self.max_speed,
            "default_altitude": self.default_altitude,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatrolRoute":
        return cls(
            id=data["id"],
            name=data["name"],
            waypoints=[PatrolWaypoint.from_dict(wp) for wp in data["waypoints"]],
            return_to_home=data.get("return_to_home", True),
            max_speed=data.get("max_speed", 5.0),
            default_altitude=data.get("default_altitude", 20.0),
        )


class PatrolStatus(Enum):
    """Status of a patrol mission."""
    SCHEDULED = "scheduled"
    QUEUED = "queued"
    PREPARING = "preparing"
    IN_PROGRESS = "in_progress"
    RETURNING = "returning"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class PatrolSchedule:
    """Schedule for recurring patrols."""
    id: str
    route_id: str
    enabled: bool = True
    days: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])  # 0=Monday
    times: List[dt_time] = field(default_factory=list)  # Times to run
    priority: int = 5  # 1-10, higher = more important
    weather_sensitive: bool = True  # Skip in bad weather
    min_battery: float = 50.0  # Minimum battery % to start

    def next_scheduled_time(self, after: datetime) -> Optional[datetime]:
        """Calculate next scheduled patrol time."""
        if not self.enabled or not self.times:
            return None

        current = after
        for _ in range(8):  # Check up to 8 days ahead
            if current.weekday() in self.days:
                for t in sorted(self.times):
                    scheduled = current.replace(
                        hour=t.hour,
                        minute=t.minute,
                        second=0,
                        microsecond=0,
                    )
                    if scheduled > after:
                        return scheduled
            current = current.replace(hour=0, minute=0) + timedelta(days=1)

        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "route_id": self.route_id,
            "enabled": self.enabled,
            "days": self.days,
            "times": [t.strftime("%H:%M") for t in self.times],
            "priority": self.priority,
            "weather_sensitive": self.weather_sensitive,
            "min_battery": self.min_battery,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatrolSchedule":
        return cls(
            id=data["id"],
            route_id=data["route_id"],
            enabled=data.get("enabled", True),
            days=data.get("days", [0, 1, 2, 3, 4, 5, 6]),
            times=[datetime.strptime(t, "%H:%M").time() for t in data.get("times", [])],
            priority=data.get("priority", 5),
            weather_sensitive=data.get("weather_sensitive", True),
            min_battery=data.get("min_battery", 50.0),
        )


@dataclass
class PatrolMission:
    """An instance of a patrol mission."""
    id: str
    route: PatrolRoute
    schedule_id: Optional[str] = None
    trigger: str = "scheduled"  # "scheduled", "manual", "alert"
    status: PatrolStatus = PatrolStatus.SCHEDULED
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    current_waypoint_index: int = 0
    progress: float = 0.0
    detections: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "route": self.route.to_dict(),
            "schedule_id": self.schedule_id,
            "trigger": self.trigger,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "current_waypoint_index": self.current_waypoint_index,
            "progress": self.progress,
            "detections": self.detections,
            "error": self.error,
        }


class PatrolScheduler:
    """
    Manages scheduled and on-demand drone patrols.

    Features:
    - Recurring patrol schedules
    - Priority-based mission queuing
    - Weather-aware scheduling
    - Alert-triggered patrols
    """

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.routes: Dict[str, PatrolRoute] = {}
        self.schedules: Dict[str, PatrolSchedule] = {}
        self.mission_queue: List[PatrolMission] = []
        self.active_mission: Optional[PatrolMission] = None
        self.mission_history: List[PatrolMission] = []

        self._scheduler_task: Optional[asyncio.Task] = None
        self._running = False
        self._mission_callback: Optional[Callable[[PatrolMission], Any]] = None
        self._weather_checker: Optional[Callable[[], bool]] = None
        self._battery_checker: Optional[Callable[[], float]] = None

        self._load_data()

    def _load_data(self):
        """Load routes and schedules from storage."""
        routes_file = self.storage_path / "patrol_routes.json"
        schedules_file = self.storage_path / "patrol_schedules.json"

        if routes_file.exists():
            with open(routes_file) as f:
                data = json.load(f)
                self.routes = {
                    r["id"]: PatrolRoute.from_dict(r) for r in data
                }

        if schedules_file.exists():
            with open(schedules_file) as f:
                data = json.load(f)
                self.schedules = {
                    s["id"]: PatrolSchedule.from_dict(s) for s in data
                }

    def _save_data(self):
        """Save routes and schedules to storage."""
        self.storage_path.mkdir(parents=True, exist_ok=True)

        routes_file = self.storage_path / "patrol_routes.json"
        with open(routes_file, "w") as f:
            json.dump([r.to_dict() for r in self.routes.values()], f, indent=2)

        schedules_file = self.storage_path / "patrol_schedules.json"
        with open(schedules_file, "w") as f:
            json.dump([s.to_dict() for s in self.schedules.values()], f, indent=2)

    def set_mission_callback(self, callback: Callable[[PatrolMission], Any]):
        """Set callback for mission execution."""
        self._mission_callback = callback

    def set_weather_checker(self, checker: Callable[[], bool]):
        """Set callback to check if weather is suitable for patrol."""
        self._weather_checker = checker

    def set_battery_checker(self, checker: Callable[[], float]):
        """Set callback to check drone battery level."""
        self._battery_checker = checker

    # Route management
    def add_route(self, route: PatrolRoute):
        """Add or update a patrol route."""
        self.routes[route.id] = route
        self._save_data()
        logger.info(f"Added route: {route.name}")

    def delete_route(self, route_id: str):
        """Delete a patrol route."""
        if route_id in self.routes:
            del self.routes[route_id]
            self._save_data()
            logger.info(f"Deleted route: {route_id}")

    def get_route(self, route_id: str) -> Optional[PatrolRoute]:
        """Get a route by ID."""
        return self.routes.get(route_id)

    def list_routes(self) -> List[PatrolRoute]:
        """List all routes."""
        return list(self.routes.values())

    # Schedule management
    def add_schedule(self, schedule: PatrolSchedule):
        """Add or update a patrol schedule."""
        if schedule.route_id not in self.routes:
            raise ValueError(f"Route {schedule.route_id} not found")
        self.schedules[schedule.id] = schedule
        self._save_data()
        logger.info(f"Added schedule for route: {schedule.route_id}")

    def delete_schedule(self, schedule_id: str):
        """Delete a patrol schedule."""
        if schedule_id in self.schedules:
            del self.schedules[schedule_id]
            self._save_data()
            logger.info(f"Deleted schedule: {schedule_id}")

    def get_schedule(self, schedule_id: str) -> Optional[PatrolSchedule]:
        """Get a schedule by ID."""
        return self.schedules.get(schedule_id)

    def list_schedules(self) -> List[PatrolSchedule]:
        """List all schedules."""
        return list(self.schedules.values())

    # Mission management
    def queue_mission(
        self,
        route_id: str,
        trigger: str = "manual",
        priority: int = 5,
    ) -> PatrolMission:
        """Queue a patrol mission."""
        route = self.routes.get(route_id)
        if not route:
            raise ValueError(f"Route {route_id} not found")

        import uuid
        mission = PatrolMission(
            id=f"mission-{uuid.uuid4().hex[:8]}",
            route=route,
            trigger=trigger,
            status=PatrolStatus.QUEUED,
        )

        # Insert based on priority
        insert_idx = len(self.mission_queue)
        for i, m in enumerate(self.mission_queue):
            if priority > 5:  # Higher priority, insert earlier
                insert_idx = i
                break

        self.mission_queue.insert(insert_idx, mission)
        logger.info(f"Queued mission: {mission.id} for route {route.name}")

        return mission

    def cancel_mission(self, mission_id: str) -> bool:
        """Cancel a queued or active mission."""
        # Check queue
        for i, mission in enumerate(self.mission_queue):
            if mission.id == mission_id:
                mission.status = PatrolStatus.CANCELLED
                self.mission_queue.pop(i)
                self.mission_history.append(mission)
                logger.info(f"Cancelled queued mission: {mission_id}")
                return True

        # Check active
        if self.active_mission and self.active_mission.id == mission_id:
            self.active_mission.status = PatrolStatus.CANCELLED
            logger.info(f"Cancelled active mission: {mission_id}")
            return True

        return False

    def get_mission_status(self, mission_id: str) -> Optional[PatrolMission]:
        """Get status of a mission."""
        if self.active_mission and self.active_mission.id == mission_id:
            return self.active_mission

        for mission in self.mission_queue:
            if mission.id == mission_id:
                return mission

        for mission in self.mission_history:
            if mission.id == mission_id:
                return mission

        return None

    def trigger_alert_patrol(
        self,
        route_id: str,
        alert_data: Dict[str, Any],
    ) -> PatrolMission:
        """Trigger an immediate patrol in response to an alert."""
        mission = self.queue_mission(route_id, trigger="alert", priority=10)
        mission.route.action_params = alert_data
        logger.info(f"Triggered alert patrol: {mission.id}")
        return mission

    # Scheduler
    async def start(self):
        """Start the patrol scheduler."""
        if self._running:
            return

        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Patrol scheduler started")

    async def stop(self):
        """Stop the patrol scheduler."""
        self._running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            self._scheduler_task = None
        logger.info("Patrol scheduler stopped")

    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self._running:
            try:
                # Check for scheduled patrols
                await self._check_schedules()

                # Execute queued missions
                await self._execute_next_mission()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")

            await asyncio.sleep(30)  # Check every 30 seconds

    async def _check_schedules(self):
        """Check if any scheduled patrols should run."""
        now = datetime.now()

        for schedule in self.schedules.values():
            if not schedule.enabled:
                continue

            next_time = schedule.next_scheduled_time(now - timedelta(minutes=1))
            if next_time and abs((next_time - now).total_seconds()) < 60:
                # Time to run this schedule
                if self._should_run_patrol(schedule):
                    self.queue_mission(
                        route_id=schedule.route_id,
                        trigger="scheduled",
                        priority=schedule.priority,
                    )
                    logger.info(f"Scheduled patrol triggered: {schedule.id}")

    def _should_run_patrol(self, schedule: PatrolSchedule) -> bool:
        """Check if conditions are suitable for patrol."""
        # Check weather
        if schedule.weather_sensitive and self._weather_checker:
            if not self._weather_checker():
                logger.info("Patrol skipped due to weather")
                return False

        # Check battery
        if self._battery_checker:
            battery = self._battery_checker()
            if battery < schedule.min_battery:
                logger.info(f"Patrol skipped due to low battery: {battery}%")
                return False

        return True

    async def _execute_next_mission(self):
        """Execute the next mission in queue."""
        if self.active_mission:
            return  # Already running a mission

        if not self.mission_queue:
            return  # No missions queued

        if not self._mission_callback:
            logger.warning("No mission callback set")
            return

        mission = self.mission_queue.pop(0)
        self.active_mission = mission

        try:
            mission.status = PatrolStatus.PREPARING
            mission.started_at = datetime.now().timestamp()

            # Execute via callback (drone controller integration)
            await self._mission_callback(mission)

            mission.status = PatrolStatus.COMPLETED
            mission.completed_at = datetime.now().timestamp()
            logger.info(f"Mission completed: {mission.id}")

        except Exception as e:
            mission.status = PatrolStatus.FAILED
            mission.error = str(e)
            mission.completed_at = datetime.now().timestamp()
            logger.error(f"Mission failed: {mission.id} - {e}")

        finally:
            self.mission_history.append(mission)
            self.active_mission = None

            # Limit history size
            if len(self.mission_history) > 100:
                self.mission_history = self.mission_history[-100:]
