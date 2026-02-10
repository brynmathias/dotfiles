"""
Drone controller interface for patrol missions.

Provides abstraction for communicating with drones via
MAVLink, DJI SDK, or other protocols.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, Awaitable, List
from enum import Enum
import asyncio
import logging
import time

from .patrol_scheduler import PatrolMission, PatrolWaypoint, PatrolStatus

logger = logging.getLogger(__name__)


class DroneState(Enum):
    """Drone flight state."""
    DISCONNECTED = "disconnected"
    IDLE = "idle"
    ARMING = "arming"
    TAKING_OFF = "taking_off"
    FLYING = "flying"
    HOVERING = "hovering"
    LANDING = "landing"
    RETURNING = "returning"
    EMERGENCY = "emergency"


class DroneCommand(Enum):
    """Commands that can be sent to drone."""
    ARM = "arm"
    DISARM = "disarm"
    TAKEOFF = "takeoff"
    LAND = "land"
    GOTO = "goto"
    RETURN_HOME = "return_home"
    EMERGENCY_STOP = "emergency_stop"
    PAUSE = "pause"
    RESUME = "resume"


@dataclass
class DronePosition:
    """Current drone position."""
    lat: float
    lng: float
    altitude: float  # meters above ground
    heading: float  # degrees
    speed: float = 0.0  # m/s
    timestamp: float = field(default_factory=time.time)

    def distance_to(self, other: "DronePosition") -> float:
        """Calculate distance to another position in meters."""
        import math
        R = 6371000  # Earth radius
        phi1, phi2 = math.radians(self.lat), math.radians(other.lat)
        dphi = math.radians(other.lat - self.lat)
        dlambda = math.radians(other.lng - self.lng)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        horiz_dist = 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        vert_dist = abs(self.altitude - other.altitude)
        return math.sqrt(horiz_dist ** 2 + vert_dist ** 2)


@dataclass
class DroneStatus:
    """Complete drone status."""
    state: DroneState
    position: Optional[DronePosition] = None
    battery_percent: float = 0.0
    battery_voltage: float = 0.0
    gps_satellites: int = 0
    gps_fix: bool = False
    signal_strength: float = 0.0
    home_position: Optional[DronePosition] = None
    flight_time: float = 0.0  # seconds
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "position": {
                "lat": self.position.lat,
                "lng": self.position.lng,
                "altitude": self.position.altitude,
                "heading": self.position.heading,
                "speed": self.position.speed,
            } if self.position else None,
            "battery_percent": self.battery_percent,
            "battery_voltage": self.battery_voltage,
            "gps_satellites": self.gps_satellites,
            "gps_fix": self.gps_fix,
            "signal_strength": self.signal_strength,
            "flight_time": self.flight_time,
            "error": self.error,
        }


# Callback types
PositionCallback = Callable[[DronePosition], Awaitable[None]]
StateCallback = Callable[[DroneState], Awaitable[None]]
DetectionCallback = Callable[[Dict[str, Any]], Awaitable[None]]


class DroneController:
    """
    Abstract controller for drone operations.

    Supports:
    - MAVLink (PX4/ArduPilot)
    - DJI Mobile SDK (via bridge)
    - Simulation mode for testing
    """

    # Safety limits
    MAX_ALTITUDE = 120.0  # meters
    MIN_BATTERY_TO_FLY = 30.0  # percent
    MIN_BATTERY_TO_CONTINUE = 20.0  # percent
    MAX_DISTANCE_FROM_HOME = 1000.0  # meters
    WAYPOINT_REACHED_THRESHOLD = 2.0  # meters

    def __init__(
        self,
        drone_id: str,
        connection_string: Optional[str] = None,
        simulation: bool = False,
    ):
        self.drone_id = drone_id
        self.connection_string = connection_string
        self.simulation = simulation

        self.status = DroneStatus(state=DroneState.DISCONNECTED)
        self.current_mission: Optional[PatrolMission] = None

        self._position_callbacks: List[PositionCallback] = []
        self._state_callbacks: List[StateCallback] = []
        self._detection_callbacks: List[DetectionCallback] = []

        self._connected = False
        self._telemetry_task: Optional[asyncio.Task] = None
        self._mission_task: Optional[asyncio.Task] = None

    def add_position_callback(self, callback: PositionCallback):
        """Add callback for position updates."""
        self._position_callbacks.append(callback)

    def add_state_callback(self, callback: StateCallback):
        """Add callback for state changes."""
        self._state_callbacks.append(callback)

    def add_detection_callback(self, callback: DetectionCallback):
        """Add callback for aerial detections."""
        self._detection_callbacks.append(callback)

    async def _notify_position(self, position: DronePosition):
        """Notify position callbacks."""
        for callback in self._position_callbacks:
            try:
                await callback(position)
            except Exception as e:
                logger.error(f"Position callback error: {e}")

    async def _notify_state(self, state: DroneState):
        """Notify state callbacks."""
        for callback in self._state_callbacks:
            try:
                await callback(state)
            except Exception as e:
                logger.error(f"State callback error: {e}")

    async def _notify_detection(self, detection: Dict[str, Any]):
        """Notify detection callbacks."""
        for callback in self._detection_callbacks:
            try:
                await callback(detection)
            except Exception as e:
                logger.error(f"Detection callback error: {e}")

    async def connect(self) -> bool:
        """Connect to the drone."""
        if self._connected:
            return True

        try:
            if self.simulation:
                logger.info("Connecting to simulated drone")
                await self._connect_simulation()
            else:
                logger.info(f"Connecting to drone: {self.connection_string}")
                await self._connect_real()

            self._connected = True
            self.status.state = DroneState.IDLE
            await self._notify_state(DroneState.IDLE)

            # Start telemetry updates
            self._telemetry_task = asyncio.create_task(self._telemetry_loop())

            return True

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.status.error = str(e)
            return False

    async def disconnect(self):
        """Disconnect from the drone."""
        if self._telemetry_task:
            self._telemetry_task.cancel()
            self._telemetry_task = None

        self._connected = False
        self.status.state = DroneState.DISCONNECTED
        await self._notify_state(DroneState.DISCONNECTED)
        logger.info("Disconnected from drone")

    async def _connect_simulation(self):
        """Connect to simulated drone."""
        await asyncio.sleep(0.5)  # Simulate connection time
        self.status.battery_percent = 85.0
        self.status.battery_voltage = 16.4
        self.status.gps_satellites = 12
        self.status.gps_fix = True
        self.status.signal_strength = 0.95
        self.status.position = DronePosition(
            lat=0.0,
            lng=0.0,
            altitude=0.0,
            heading=0.0,
        )
        self.status.home_position = self.status.position

    async def _connect_real(self):
        """Connect to real drone via MAVLink or SDK."""
        # Placeholder for actual implementation
        # Would use pymavlink or DJI SDK bridge here
        raise NotImplementedError("Real drone connection not implemented")

    async def _telemetry_loop(self):
        """Loop to update telemetry data."""
        while self._connected:
            try:
                if self.simulation:
                    await self._update_simulation_telemetry()
                else:
                    await self._update_real_telemetry()

                if self.status.position:
                    await self._notify_position(self.status.position)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Telemetry error: {e}")

            await asyncio.sleep(0.1)  # 10 Hz update rate

    async def _update_simulation_telemetry(self):
        """Update telemetry for simulation mode."""
        # Simulate battery drain
        if self.status.state in [DroneState.FLYING, DroneState.HOVERING]:
            self.status.battery_percent -= 0.001
            self.status.flight_time += 0.1

    async def _update_real_telemetry(self):
        """Update telemetry from real drone."""
        pass  # Would read from MAVLink/SDK

    async def send_command(self, command: DroneCommand, **kwargs) -> bool:
        """Send a command to the drone."""
        if not self._connected:
            logger.error("Cannot send command: not connected")
            return False

        logger.info(f"Sending command: {command.value}")

        try:
            if command == DroneCommand.ARM:
                return await self._arm()
            elif command == DroneCommand.DISARM:
                return await self._disarm()
            elif command == DroneCommand.TAKEOFF:
                altitude = kwargs.get("altitude", 10.0)
                return await self._takeoff(altitude)
            elif command == DroneCommand.LAND:
                return await self._land()
            elif command == DroneCommand.GOTO:
                return await self._goto(
                    lat=kwargs["lat"],
                    lng=kwargs["lng"],
                    altitude=kwargs.get("altitude", self.status.position.altitude if self.status.position else 10.0),
                )
            elif command == DroneCommand.RETURN_HOME:
                return await self._return_home()
            elif command == DroneCommand.EMERGENCY_STOP:
                return await self._emergency_stop()
            else:
                logger.warning(f"Unknown command: {command}")
                return False

        except Exception as e:
            logger.error(f"Command failed: {e}")
            self.status.error = str(e)
            return False

    async def _arm(self) -> bool:
        """Arm the drone motors."""
        if self.status.battery_percent < self.MIN_BATTERY_TO_FLY:
            logger.error("Battery too low to arm")
            return False

        self.status.state = DroneState.ARMING
        await self._notify_state(DroneState.ARMING)

        if self.simulation:
            await asyncio.sleep(1.0)

        self.status.state = DroneState.IDLE
        await self._notify_state(DroneState.IDLE)
        return True

    async def _disarm(self) -> bool:
        """Disarm the drone motors."""
        self.status.state = DroneState.IDLE
        await self._notify_state(DroneState.IDLE)
        return True

    async def _takeoff(self, altitude: float) -> bool:
        """Take off to specified altitude."""
        altitude = min(altitude, self.MAX_ALTITUDE)

        self.status.state = DroneState.TAKING_OFF
        await self._notify_state(DroneState.TAKING_OFF)

        if self.simulation:
            # Simulate takeoff
            current_alt = 0.0
            while current_alt < altitude:
                current_alt += 0.5
                if self.status.position:
                    self.status.position.altitude = current_alt
                await asyncio.sleep(0.1)

        self.status.state = DroneState.HOVERING
        await self._notify_state(DroneState.HOVERING)
        return True

    async def _land(self) -> bool:
        """Land the drone."""
        self.status.state = DroneState.LANDING
        await self._notify_state(DroneState.LANDING)

        if self.simulation and self.status.position:
            while self.status.position.altitude > 0.1:
                self.status.position.altitude -= 0.3
                await asyncio.sleep(0.1)
            self.status.position.altitude = 0.0

        self.status.state = DroneState.IDLE
        await self._notify_state(DroneState.IDLE)
        return True

    async def _goto(self, lat: float, lng: float, altitude: float) -> bool:
        """Fly to specified position."""
        self.status.state = DroneState.FLYING
        await self._notify_state(DroneState.FLYING)

        target = DronePosition(lat=lat, lng=lng, altitude=altitude, heading=0)

        if self.simulation and self.status.position:
            # Simulate flight
            while self.status.position.distance_to(target) > self.WAYPOINT_REACHED_THRESHOLD:
                # Move towards target
                dlat = (target.lat - self.status.position.lat) * 0.1
                dlng = (target.lng - self.status.position.lng) * 0.1
                dalt = (target.altitude - self.status.position.altitude) * 0.1

                self.status.position.lat += dlat
                self.status.position.lng += dlng
                self.status.position.altitude += dalt

                await asyncio.sleep(0.1)

        self.status.state = DroneState.HOVERING
        await self._notify_state(DroneState.HOVERING)
        return True

    async def _return_home(self) -> bool:
        """Return to home position."""
        if not self.status.home_position:
            logger.error("No home position set")
            return False

        self.status.state = DroneState.RETURNING
        await self._notify_state(DroneState.RETURNING)

        await self._goto(
            lat=self.status.home_position.lat,
            lng=self.status.home_position.lng,
            altitude=self.status.home_position.altitude + 10,  # Safety altitude
        )

        await self._land()
        return True

    async def _emergency_stop(self) -> bool:
        """Emergency stop - immediate landing."""
        self.status.state = DroneState.EMERGENCY
        await self._notify_state(DroneState.EMERGENCY)
        await self._land()
        return True

    async def execute_mission(self, mission: PatrolMission) -> bool:
        """
        Execute a patrol mission.

        Args:
            mission: The mission to execute

        Returns:
            True if mission completed successfully
        """
        if not self._connected:
            logger.error("Cannot execute mission: not connected")
            return False

        self.current_mission = mission
        mission.status = PatrolStatus.IN_PROGRESS

        try:
            # Arm and takeoff
            if not await self._arm():
                raise Exception("Failed to arm")

            altitude = mission.route.default_altitude
            if not await self._takeoff(altitude):
                raise Exception("Failed to takeoff")

            # Execute waypoints
            for i, waypoint in enumerate(mission.route.waypoints):
                mission.current_waypoint_index = i
                mission.progress = i / len(mission.route.waypoints)

                # Check battery
                if self.status.battery_percent < self.MIN_BATTERY_TO_CONTINUE:
                    logger.warning("Low battery, returning home")
                    break

                # Check if cancelled
                if mission.status == PatrolStatus.CANCELLED:
                    break

                # Fly to waypoint
                await self._goto(
                    lat=waypoint.lat,
                    lng=waypoint.lng,
                    altitude=waypoint.altitude,
                )

                # Execute waypoint action
                if waypoint.action:
                    await self._execute_waypoint_action(waypoint, mission)

                # Hover time
                if waypoint.hover_time > 0:
                    await asyncio.sleep(waypoint.hover_time)

            # Return home if configured
            if mission.route.return_to_home:
                mission.status = PatrolStatus.RETURNING
                await self._return_home()

            mission.progress = 1.0
            return True

        except Exception as e:
            logger.error(f"Mission execution error: {e}")
            mission.error = str(e)
            await self._return_home()  # Try to return home on error
            return False

        finally:
            self.current_mission = None

    async def _execute_waypoint_action(
        self,
        waypoint: PatrolWaypoint,
        mission: PatrolMission,
    ):
        """Execute action at a waypoint."""
        action = waypoint.action

        if action == "photo":
            # Simulate taking photo
            logger.info(f"Taking photo at {waypoint.id}")
            await asyncio.sleep(0.5)

        elif action == "video":
            duration = waypoint.action_params.get("duration", 5.0)
            logger.info(f"Recording video for {duration}s at {waypoint.id}")
            await asyncio.sleep(duration)

        elif action == "scan":
            # Simulate area scan for predators
            logger.info(f"Scanning area at {waypoint.id}")
            await asyncio.sleep(2.0)

            # Simulated detection (would use onboard camera/AI)
            if self.simulation:
                import random
                if random.random() < 0.1:  # 10% chance of detection
                    detection = {
                        "type": "fox",
                        "confidence": random.uniform(0.7, 0.95),
                        "position": {
                            "lat": waypoint.lat + random.uniform(-0.0001, 0.0001),
                            "lng": waypoint.lng + random.uniform(-0.0001, 0.0001),
                        },
                        "timestamp": time.time(),
                    }
                    mission.detections.append(detection)
                    await self._notify_detection(detection)

    def get_status(self) -> DroneStatus:
        """Get current drone status."""
        return self.status

    def is_flying(self) -> bool:
        """Check if drone is currently in flight."""
        return self.status.state in [
            DroneState.FLYING,
            DroneState.HOVERING,
            DroneState.TAKING_OFF,
            DroneState.LANDING,
            DroneState.RETURNING,
        ]
