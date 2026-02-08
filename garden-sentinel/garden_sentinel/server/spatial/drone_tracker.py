"""
Drone camera tracking and flight path management.

Handles mobile cameras including drones with:
- Real-time position streaming
- Flight path execution tracking
- GPS to local coordinate conversion
- Gimbal orientation tracking
"""

import time
import math
import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable


@dataclass
class GPSCoordinate:
    """GPS coordinate with altitude."""
    latitude: float
    longitude: float
    altitude: float = 0.0

    def to_dict(self) -> dict:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude,
        }


@dataclass
class GimbalState:
    """Gimbal orientation state."""
    pitch: float = 0.0    # Degrees, 0 = horizontal, negative = looking down
    roll: float = 0.0     # Degrees
    yaw: float = 0.0      # Degrees, 0 = north


class DroneStatus(Enum):
    """Drone operational status."""
    OFFLINE = "offline"
    IDLE = "idle"              # On ground, ready
    TAKING_OFF = "taking_off"
    FLYING = "flying"
    LANDING = "landing"
    RETURNING_HOME = "returning_home"
    LOW_BATTERY = "low_battery"
    EMERGENCY = "emergency"


@dataclass
class DroneTelemetry:
    """Complete drone telemetry snapshot."""
    timestamp: float

    # Position
    gps: GPSCoordinate
    local_position: Optional[tuple[float, float, float]] = None  # x, y, z in local coords

    # Orientation
    heading: float = 0.0      # Degrees from north
    gimbal: Optional[GimbalState] = None

    # Flight data
    speed: float = 0.0        # m/s ground speed
    vertical_speed: float = 0.0  # m/s, positive = ascending
    altitude_agl: float = 0.0    # Altitude above ground level

    # Status
    battery_percent: float = 100.0
    gps_fix_quality: int = 0     # 0=no fix, 3=3D fix, 4=RTK
    satellite_count: int = 0

    # Flight path
    current_waypoint_index: Optional[int] = None
    distance_to_waypoint: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "gps": self.gps.to_dict(),
            "local_position": self.local_position,
            "heading": self.heading,
            "gimbal": {
                "pitch": self.gimbal.pitch,
                "roll": self.gimbal.roll,
                "yaw": self.gimbal.yaw,
            } if self.gimbal else None,
            "speed": self.speed,
            "vertical_speed": self.vertical_speed,
            "altitude_agl": self.altitude_agl,
            "battery_percent": self.battery_percent,
            "gps_fix_quality": self.gps_fix_quality,
            "satellite_count": self.satellite_count,
            "current_waypoint_index": self.current_waypoint_index,
            "distance_to_waypoint": self.distance_to_waypoint,
        }


@dataclass
class MobileCamera:
    """Tracks a mobile camera (drone, vehicle, handheld)."""
    camera_id: str
    camera_type: str  # "drone", "vehicle", "handheld"
    status: DroneStatus = DroneStatus.OFFLINE

    # Current state
    current_telemetry: Optional[DroneTelemetry] = None
    telemetry_history: list[DroneTelemetry] = field(default_factory=list)

    # Flight path tracking
    assigned_path_id: Optional[str] = None
    path_start_time: Optional[float] = None
    is_executing_path: bool = False

    # Capabilities
    max_speed: float = 15.0           # m/s
    max_altitude: float = 120.0       # meters
    min_battery_for_return: float = 20.0  # percent

    # History settings
    max_history_size: int = 1000

    def update_telemetry(self, telemetry: DroneTelemetry):
        """Update with new telemetry."""
        self.current_telemetry = telemetry
        self.telemetry_history.append(telemetry)

        # Trim history
        if len(self.telemetry_history) > self.max_history_size:
            self.telemetry_history = self.telemetry_history[-self.max_history_size:]

    def get_trajectory(self, time_window: float = 300.0) -> list[tuple[float, float]]:
        """Get recent trajectory as list of (x, y) positions."""
        now = time.time()
        return [
            (t.local_position[0], t.local_position[1])
            for t in self.telemetry_history
            if t.local_position and now - t.timestamp <= time_window
        ]


class GPSConverter:
    """
    Converts GPS coordinates to local coordinate system.

    Uses a local tangent plane approximation centered on an origin point.
    """

    # Earth radius in meters
    EARTH_RADIUS = 6371000

    def __init__(self, origin: GPSCoordinate):
        """Initialize with origin point for local coordinate system."""
        self.origin = origin
        self._lat_rad = math.radians(origin.latitude)
        self._lon_rad = math.radians(origin.longitude)

        # Meters per degree at this latitude
        self._meters_per_lat = self.EARTH_RADIUS * math.pi / 180
        self._meters_per_lon = self.EARTH_RADIUS * math.cos(self._lat_rad) * math.pi / 180

    def to_local(self, gps: GPSCoordinate) -> tuple[float, float, float]:
        """
        Convert GPS to local coordinates (x, y, z) in meters.

        x = east, y = north, z = altitude
        """
        x = (gps.longitude - self.origin.longitude) * self._meters_per_lon
        y = (gps.latitude - self.origin.latitude) * self._meters_per_lat
        z = gps.altitude - self.origin.altitude

        return (x, y, z)

    def to_gps(self, x: float, y: float, z: float = 0.0) -> GPSCoordinate:
        """Convert local coordinates to GPS."""
        lat = self.origin.latitude + y / self._meters_per_lat
        lon = self.origin.longitude + x / self._meters_per_lon
        alt = self.origin.altitude + z

        return GPSCoordinate(lat, lon, alt)


class DroneTracker:
    """
    Tracks multiple drone cameras with real-time position updates.

    Handles:
    - Position streaming from drones
    - GPS to local coordinate conversion
    - Flight path execution tracking
    - Coverage calculation for moving cameras
    """

    def __init__(self, gps_converter: Optional[GPSConverter] = None):
        self.gps_converter = gps_converter
        self.drones: dict[str, MobileCamera] = {}

        # Callbacks
        self._position_callbacks: list[Callable[[str, DroneTelemetry], None]] = []
        self._status_callbacks: list[Callable[[str, DroneStatus], None]] = []

    def register_drone(
        self,
        camera_id: str,
        camera_type: str = "drone",
        max_speed: float = 15.0,
        max_altitude: float = 120.0,
    ) -> MobileCamera:
        """Register a new mobile camera."""
        drone = MobileCamera(
            camera_id=camera_id,
            camera_type=camera_type,
            max_speed=max_speed,
            max_altitude=max_altitude,
        )
        self.drones[camera_id] = drone
        return drone

    def unregister_drone(self, camera_id: str):
        """Remove a drone from tracking."""
        if camera_id in self.drones:
            del self.drones[camera_id]

    def update_position(
        self,
        camera_id: str,
        gps: GPSCoordinate,
        heading: float = 0.0,
        gimbal: Optional[GimbalState] = None,
        speed: float = 0.0,
        vertical_speed: float = 0.0,
        battery_percent: float = 100.0,
        gps_fix_quality: int = 3,
        satellite_count: int = 0,
        timestamp: Optional[float] = None,
    ):
        """
        Update drone position from telemetry.

        Call this method with each position update from the drone.
        """
        if camera_id not in self.drones:
            self.register_drone(camera_id)

        drone = self.drones[camera_id]
        timestamp = timestamp or time.time()

        # Convert GPS to local if converter available
        local_position = None
        if self.gps_converter:
            local_position = self.gps_converter.to_local(gps)

        telemetry = DroneTelemetry(
            timestamp=timestamp,
            gps=gps,
            local_position=local_position,
            heading=heading,
            gimbal=gimbal,
            speed=speed,
            vertical_speed=vertical_speed,
            altitude_agl=gps.altitude if not self.gps_converter else local_position[2],
            battery_percent=battery_percent,
            gps_fix_quality=gps_fix_quality,
            satellite_count=satellite_count,
        )

        # Update drone state
        old_status = drone.status
        drone.update_telemetry(telemetry)

        # Update status based on telemetry
        if battery_percent < drone.min_battery_for_return:
            drone.status = DroneStatus.LOW_BATTERY
        elif speed > 0.5 or abs(vertical_speed) > 0.5:
            drone.status = DroneStatus.FLYING
        else:
            drone.status = DroneStatus.IDLE

        # Trigger callbacks
        for callback in self._position_callbacks:
            try:
                callback(camera_id, telemetry)
            except Exception:
                pass

        if old_status != drone.status:
            for callback in self._status_callbacks:
                try:
                    callback(camera_id, drone.status)
                except Exception:
                    pass

    def update_path_progress(
        self,
        camera_id: str,
        waypoint_index: int,
        distance_to_waypoint: float,
    ):
        """Update flight path execution progress."""
        if camera_id not in self.drones:
            return

        drone = self.drones[camera_id]
        if drone.current_telemetry:
            drone.current_telemetry.current_waypoint_index = waypoint_index
            drone.current_telemetry.distance_to_waypoint = distance_to_waypoint

    def assign_flight_path(self, camera_id: str, path_id: str):
        """Assign a flight path to a drone."""
        if camera_id in self.drones:
            drone = self.drones[camera_id]
            drone.assigned_path_id = path_id
            drone.path_start_time = time.time()
            drone.is_executing_path = True

    def complete_flight_path(self, camera_id: str):
        """Mark flight path as completed."""
        if camera_id in self.drones:
            drone = self.drones[camera_id]
            drone.is_executing_path = False

    def get_drone(self, camera_id: str) -> Optional[MobileCamera]:
        """Get a drone by ID."""
        return self.drones.get(camera_id)

    def get_all_drones(self) -> list[MobileCamera]:
        """Get all registered drones."""
        return list(self.drones.values())

    def get_active_drones(self) -> list[MobileCamera]:
        """Get drones that are currently flying."""
        return [
            d for d in self.drones.values()
            if d.status in (DroneStatus.FLYING, DroneStatus.TAKING_OFF, DroneStatus.LANDING)
        ]

    def get_coverage_for_drone(
        self,
        camera_id: str,
        fov: float = 90.0,
        range_m: float = 30.0,
    ) -> Optional[list[tuple[float, float]]]:
        """
        Calculate current coverage polygon for a drone camera.

        Returns list of (x, y) points forming coverage polygon, or None if drone not found.
        """
        drone = self.drones.get(camera_id)
        if not drone or not drone.current_telemetry or not drone.current_telemetry.local_position:
            return None

        pos = drone.current_telemetry.local_position
        heading = drone.current_telemetry.heading

        # Adjust heading for gimbal yaw if available
        if drone.current_telemetry.gimbal:
            heading += drone.current_telemetry.gimbal.yaw

        cx, cy = pos[0], pos[1]

        # Calculate cone edges
        half_fov = fov / 2
        left_angle = math.radians(heading - half_fov)
        right_angle = math.radians(heading + half_fov)

        # Generate arc points
        arc_points = []
        num_points = 20
        for i in range(num_points + 1):
            angle = left_angle + (right_angle - left_angle) * i / num_points
            x = cx + range_m * math.sin(angle)
            y = cy + range_m * math.cos(angle)
            arc_points.append((x, y))

        # Build polygon: camera position -> arc
        polygon = [(cx, cy)] + arc_points

        return polygon

    def register_position_callback(self, callback: Callable[[str, DroneTelemetry], None]):
        """Register callback for position updates."""
        self._position_callbacks.append(callback)

    def register_status_callback(self, callback: Callable[[str, DroneStatus], None]):
        """Register callback for status changes."""
        self._status_callbacks.append(callback)

    def to_dict(self) -> dict:
        """Export tracker state as dictionary."""
        return {
            "drones": {
                camera_id: {
                    "camera_id": drone.camera_id,
                    "camera_type": drone.camera_type,
                    "status": drone.status.value,
                    "telemetry": drone.current_telemetry.to_dict() if drone.current_telemetry else None,
                    "assigned_path_id": drone.assigned_path_id,
                    "is_executing_path": drone.is_executing_path,
                }
                for camera_id, drone in self.drones.items()
            }
        }


class DronePositionReceiver:
    """
    Handles incoming position streams from drones.

    Can be extended to support different protocols:
    - MAVLink
    - DJI SDK
    - Custom UDP/TCP
    """

    def __init__(self, tracker: DroneTracker):
        self.tracker = tracker
        self._running = False

    async def handle_mavlink_message(self, camera_id: str, message: dict):
        """
        Handle a MAVLink-style position message.

        Expected message format:
        {
            "lat": int (degE7),
            "lon": int (degE7),
            "alt": int (mm AMSL),
            "relative_alt": int (mm AGL),
            "vx": int (cm/s),
            "vy": int (cm/s),
            "vz": int (cm/s),
            "hdg": int (cdeg, 0-35999)
        }
        """
        gps = GPSCoordinate(
            latitude=message["lat"] / 1e7,
            longitude=message["lon"] / 1e7,
            altitude=message.get("relative_alt", message["alt"]) / 1000.0,
        )

        # Calculate ground speed from vx, vy
        vx = message.get("vx", 0) / 100.0  # cm/s to m/s
        vy = message.get("vy", 0) / 100.0
        vz = message.get("vz", 0) / 100.0
        speed = math.sqrt(vx**2 + vy**2)

        heading = message.get("hdg", 0) / 100.0  # cdeg to deg

        self.tracker.update_position(
            camera_id=camera_id,
            gps=gps,
            heading=heading,
            speed=speed,
            vertical_speed=-vz,  # MAVLink uses NED, positive Z is down
        )

    async def handle_dji_telemetry(self, camera_id: str, telemetry: dict):
        """
        Handle DJI-style telemetry message.

        Expected format:
        {
            "latitude": float (degrees),
            "longitude": float (degrees),
            "altitude": float (meters),
            "compass_heading": float (degrees),
            "gimbal_pitch": float (degrees),
            "gimbal_yaw": float (degrees),
            "velocity_x": float (m/s),
            "velocity_y": float (m/s),
            "velocity_z": float (m/s),
            "battery_level": int (percent),
            "satellite_count": int
        }
        """
        gps = GPSCoordinate(
            latitude=telemetry["latitude"],
            longitude=telemetry["longitude"],
            altitude=telemetry["altitude"],
        )

        gimbal = GimbalState(
            pitch=telemetry.get("gimbal_pitch", 0.0),
            yaw=telemetry.get("gimbal_yaw", 0.0),
        )

        vx = telemetry.get("velocity_x", 0.0)
        vy = telemetry.get("velocity_y", 0.0)
        speed = math.sqrt(vx**2 + vy**2)

        self.tracker.update_position(
            camera_id=camera_id,
            gps=gps,
            heading=telemetry.get("compass_heading", 0.0),
            gimbal=gimbal,
            speed=speed,
            vertical_speed=telemetry.get("velocity_z", 0.0),
            battery_percent=telemetry.get("battery_level", 100.0),
            satellite_count=telemetry.get("satellite_count", 0),
        )

    async def handle_generic_position(
        self,
        camera_id: str,
        latitude: float,
        longitude: float,
        altitude: float,
        heading: float = 0.0,
        speed: float = 0.0,
        battery_percent: float = 100.0,
    ):
        """Handle a generic position update."""
        gps = GPSCoordinate(latitude, longitude, altitude)

        self.tracker.update_position(
            camera_id=camera_id,
            gps=gps,
            heading=heading,
            speed=speed,
            battery_percent=battery_percent,
        )
