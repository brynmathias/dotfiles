"""
Garden spatial configuration.

Defines:
- Garden boundaries
- Protected zones (coop, feeding areas)
- Camera placements with coverage cones
- Entry points and perimeter
"""

import math
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Point:
    """A 2D point in garden coordinates (meters from origin)."""
    x: float
    y: float

    def distance_to(self, other: "Point") -> float:
        """Calculate distance to another point."""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, data: dict) -> "Point":
        return cls(x=data["x"], y=data["y"])


@dataclass
class Polygon:
    """A polygon defined by a list of points."""
    points: list[Point]

    def contains(self, point: Point) -> bool:
        """Check if a point is inside the polygon using ray casting."""
        n = len(self.points)
        inside = False

        j = n - 1
        for i in range(n):
            xi, yi = self.points[i].x, self.points[i].y
            xj, yj = self.points[j].x, self.points[j].y

            if ((yi > point.y) != (yj > point.y)) and \
               (point.x < (xj - xi) * (point.y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i

        return inside

    def centroid(self) -> Point:
        """Calculate the centroid of the polygon."""
        cx = sum(p.x for p in self.points) / len(self.points)
        cy = sum(p.y for p in self.points) / len(self.points)
        return Point(cx, cy)

    def area(self) -> float:
        """Calculate the area of the polygon using shoelace formula."""
        n = len(self.points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.points[i].x * self.points[j].y
            area -= self.points[j].x * self.points[i].y
        return abs(area) / 2.0

    def to_dict(self) -> dict:
        return {"points": [p.to_dict() for p in self.points]}

    @classmethod
    def from_dict(cls, data: dict) -> "Polygon":
        return cls(points=[Point.from_dict(p) for p in data["points"]])

    @classmethod
    def rectangle(cls, x: float, y: float, width: float, height: float) -> "Polygon":
        """Create a rectangle polygon."""
        return cls(points=[
            Point(x, y),
            Point(x + width, y),
            Point(x + width, y + height),
            Point(x, y + height),
        ])

    @classmethod
    def circle(cls, center: Point, radius: float, segments: int = 16) -> "Polygon":
        """Create a circular polygon approximation."""
        points = []
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            points.append(Point(
                center.x + radius * math.cos(angle),
                center.y + radius * math.sin(angle),
            ))
        return cls(points=points)


class ZoneType(Enum):
    """Type of zone in the garden."""
    PROTECTED = "protected"      # Chicken coop, nesting areas
    FEEDING = "feeding"          # Feeding areas
    PERIMETER = "perimeter"      # Garden boundary
    ENTRY_POINT = "entry_point"  # Known entry points (fence gaps, etc.)
    EXCLUSION = "exclusion"      # Areas to ignore (buildings, etc.)
    PATROL = "patrol"            # Areas for drone patrols


@dataclass
class Zone:
    """A zone in the garden with a specific purpose."""
    zone_id: str
    name: str
    zone_type: ZoneType
    boundary: Polygon
    priority: int = 0  # Higher = more important to protect
    alert_on_entry: bool = True
    alert_on_exit: bool = False
    color: str = "#00ff00"  # For visualization

    def contains(self, point: Point) -> bool:
        """Check if a point is in this zone."""
        return self.boundary.contains(point)

    def to_dict(self) -> dict:
        return {
            "zone_id": self.zone_id,
            "name": self.name,
            "zone_type": self.zone_type.value,
            "boundary": self.boundary.to_dict(),
            "priority": self.priority,
            "alert_on_entry": self.alert_on_entry,
            "alert_on_exit": self.alert_on_exit,
            "color": self.color,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Zone":
        return cls(
            zone_id=data["zone_id"],
            name=data["name"],
            zone_type=ZoneType(data["zone_type"]),
            boundary=Polygon.from_dict(data["boundary"]),
            priority=data.get("priority", 0),
            alert_on_entry=data.get("alert_on_entry", True),
            alert_on_exit=data.get("alert_on_exit", False),
            color=data.get("color", "#00ff00"),
        )


@dataclass
class CameraPlacement:
    """Camera position and orientation in the garden."""
    camera_id: str
    name: str
    position: Point
    height: float = 2.0  # meters above ground
    heading: float = 0.0  # degrees, 0 = North, clockwise
    fov_horizontal: float = 62.0  # degrees
    fov_vertical: float = 49.0
    range: float = 15.0  # effective detection range in meters

    # For mobile cameras (drones)
    is_mobile: bool = False
    current_position: Optional[Point] = None
    current_heading: Optional[float] = None

    # Coverage
    has_sprayer: bool = True
    sprayer_range: float = 5.0

    def get_effective_position(self) -> Point:
        """Get the current position (for mobile cameras) or fixed position."""
        if self.is_mobile and self.current_position:
            return self.current_position
        return self.position

    def get_effective_heading(self) -> float:
        """Get the current heading (for mobile cameras) or fixed heading."""
        if self.is_mobile and self.current_heading is not None:
            return self.current_heading
        return self.heading

    def get_coverage_cone(self) -> list[Point]:
        """Get the coverage cone as a polygon for visualization."""
        pos = self.get_effective_position()
        heading = self.get_effective_heading()

        # Calculate cone points
        half_fov = math.radians(self.fov_horizontal / 2)
        heading_rad = math.radians(heading)

        # Camera position
        points = [pos]

        # Left edge of cone
        left_angle = heading_rad - half_fov
        points.append(Point(
            pos.x + self.range * math.sin(left_angle),
            pos.y + self.range * math.cos(left_angle),
        ))

        # Arc along the range
        num_arc_points = 8
        for i in range(1, num_arc_points):
            angle = left_angle + (2 * half_fov * i / num_arc_points)
            points.append(Point(
                pos.x + self.range * math.sin(angle),
                pos.y + self.range * math.cos(angle),
            ))

        # Right edge of cone
        right_angle = heading_rad + half_fov
        points.append(Point(
            pos.x + self.range * math.sin(right_angle),
            pos.y + self.range * math.cos(right_angle),
        ))

        return points

    def can_see_point(self, point: Point) -> bool:
        """Check if this camera can see a given point."""
        pos = self.get_effective_position()
        heading = self.get_effective_heading()

        # Check distance
        distance = pos.distance_to(point)
        if distance > self.range:
            return False

        # Check angle
        angle_to_point = math.degrees(math.atan2(
            point.x - pos.x,
            point.y - pos.y
        ))

        # Normalize angles
        angle_diff = abs((angle_to_point - heading + 180) % 360 - 180)

        return angle_diff <= self.fov_horizontal / 2

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "position": self.position.to_dict(),
            "height": self.height,
            "heading": self.heading,
            "fov_horizontal": self.fov_horizontal,
            "fov_vertical": self.fov_vertical,
            "range": self.range,
            "is_mobile": self.is_mobile,
            "current_position": self.current_position.to_dict() if self.current_position else None,
            "current_heading": self.current_heading,
            "has_sprayer": self.has_sprayer,
            "sprayer_range": self.sprayer_range,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CameraPlacement":
        return cls(
            camera_id=data["camera_id"],
            name=data["name"],
            position=Point.from_dict(data["position"]),
            height=data.get("height", 2.0),
            heading=data.get("heading", 0.0),
            fov_horizontal=data.get("fov_horizontal", 62.0),
            fov_vertical=data.get("fov_vertical", 49.0),
            range=data.get("range", 15.0),
            is_mobile=data.get("is_mobile", False),
            current_position=Point.from_dict(data["current_position"]) if data.get("current_position") else None,
            current_heading=data.get("current_heading"),
            has_sprayer=data.get("has_sprayer", True),
            sprayer_range=data.get("sprayer_range", 5.0),
        )


@dataclass
class FlightPath:
    """A pre-planned flight path for drones."""
    path_id: str
    name: str
    waypoints: list[Point]
    altitude: float = 10.0  # meters
    speed: float = 5.0  # m/s
    loop: bool = True
    camera_headings: list[float] = field(default_factory=list)  # Heading at each waypoint

    def get_total_distance(self) -> float:
        """Calculate total path distance."""
        if len(self.waypoints) < 2:
            return 0.0

        total = 0.0
        for i in range(len(self.waypoints) - 1):
            total += self.waypoints[i].distance_to(self.waypoints[i + 1])

        if self.loop:
            total += self.waypoints[-1].distance_to(self.waypoints[0])

        return total

    def get_estimated_duration(self) -> float:
        """Estimate flight duration in seconds."""
        return self.get_total_distance() / self.speed

    def to_dict(self) -> dict:
        return {
            "path_id": self.path_id,
            "name": self.name,
            "waypoints": [p.to_dict() for p in self.waypoints],
            "altitude": self.altitude,
            "speed": self.speed,
            "loop": self.loop,
            "camera_headings": self.camera_headings,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FlightPath":
        return cls(
            path_id=data["path_id"],
            name=data["name"],
            waypoints=[Point.from_dict(p) for p in data["waypoints"]],
            altitude=data.get("altitude", 10.0),
            speed=data.get("speed", 5.0),
            loop=data.get("loop", True),
            camera_headings=data.get("camera_headings", []),
        )


class GardenMap:
    """
    Complete garden spatial configuration.

    Stores:
    - Garden boundary
    - Protected zones
    - Camera placements
    - Flight paths for drones
    """

    def __init__(self):
        self.boundary: Optional[Polygon] = None
        self.zones: dict[str, Zone] = {}
        self.cameras: dict[str, CameraPlacement] = {}
        self.flight_paths: dict[str, FlightPath] = {}
        self.origin_lat: Optional[float] = None  # For GPS mapping
        self.origin_lon: Optional[float] = None
        self.rotation: float = 0.0  # Rotation of local coords relative to true north

    def set_boundary(self, boundary: Polygon):
        """Set the garden boundary."""
        self.boundary = boundary
        logger.info(f"Garden boundary set: {len(boundary.points)} points, {boundary.area():.1f} m²")

    def add_zone(self, zone: Zone):
        """Add a zone to the garden."""
        self.zones[zone.zone_id] = zone
        logger.info(f"Added zone: {zone.name} ({zone.zone_type.value})")

    def remove_zone(self, zone_id: str):
        """Remove a zone from the garden."""
        if zone_id in self.zones:
            del self.zones[zone_id]

    def add_camera(self, camera: CameraPlacement):
        """Add a camera to the garden."""
        self.cameras[camera.camera_id] = camera
        logger.info(f"Added camera: {camera.name} at ({camera.position.x}, {camera.position.y})")

    def remove_camera(self, camera_id: str):
        """Remove a camera from the garden."""
        if camera_id in self.cameras:
            del self.cameras[camera_id]

    def add_flight_path(self, path: FlightPath):
        """Add a flight path for drones."""
        self.flight_paths[path.path_id] = path
        logger.info(f"Added flight path: {path.name} ({len(path.waypoints)} waypoints)")

    def update_camera_position(self, camera_id: str, position: Point, heading: Optional[float] = None):
        """Update a mobile camera's position (e.g., from GPS)."""
        if camera_id in self.cameras:
            camera = self.cameras[camera_id]
            camera.current_position = position
            if heading is not None:
                camera.current_heading = heading

    def get_zones_at_point(self, point: Point) -> list[Zone]:
        """Get all zones that contain a point."""
        return [zone for zone in self.zones.values() if zone.contains(point)]

    def get_cameras_that_see_point(self, point: Point) -> list[CameraPlacement]:
        """Get all cameras that can see a point."""
        return [cam for cam in self.cameras.values() if cam.can_see_point(point)]

    def get_coverage_gaps(self) -> list[Point]:
        """Find points in the garden that aren't covered by any camera."""
        # Sample points in a grid
        gaps = []
        if not self.boundary:
            return gaps

        # Find bounding box
        min_x = min(p.x for p in self.boundary.points)
        max_x = max(p.x for p in self.boundary.points)
        min_y = min(p.y for p in self.boundary.points)
        max_y = max(p.y for p in self.boundary.points)

        # Sample at 1m intervals
        x = min_x
        while x <= max_x:
            y = min_y
            while y <= max_y:
                point = Point(x, y)
                if self.boundary.contains(point):
                    if not any(cam.can_see_point(point) for cam in self.cameras.values()):
                        gaps.append(point)
                y += 1.0
            x += 1.0

        return gaps

    def gps_to_local(self, lat: float, lon: float) -> Optional[Point]:
        """Convert GPS coordinates to local garden coordinates."""
        if self.origin_lat is None or self.origin_lon is None:
            return None

        # Approximate conversion (works for small areas)
        # 1 degree latitude ≈ 111km
        # 1 degree longitude ≈ 111km * cos(latitude)
        lat_m = (lat - self.origin_lat) * 111000
        lon_m = (lon - self.origin_lon) * 111000 * math.cos(math.radians(self.origin_lat))

        # Apply rotation
        if self.rotation != 0:
            rot_rad = math.radians(self.rotation)
            x = lat_m * math.cos(rot_rad) - lon_m * math.sin(rot_rad)
            y = lat_m * math.sin(rot_rad) + lon_m * math.cos(rot_rad)
            return Point(x, y)

        return Point(lon_m, lat_m)

    def local_to_gps(self, point: Point) -> Optional[tuple[float, float]]:
        """Convert local coordinates to GPS."""
        if self.origin_lat is None or self.origin_lon is None:
            return None

        # Reverse rotation
        x, y = point.x, point.y
        if self.rotation != 0:
            rot_rad = -math.radians(self.rotation)
            x = point.x * math.cos(rot_rad) - point.y * math.sin(rot_rad)
            y = point.x * math.sin(rot_rad) + point.y * math.cos(rot_rad)

        lat = self.origin_lat + (y / 111000)
        lon = self.origin_lon + (x / (111000 * math.cos(math.radians(self.origin_lat))))

        return (lat, lon)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "boundary": self.boundary.to_dict() if self.boundary else None,
            "zones": {k: v.to_dict() for k, v in self.zones.items()},
            "cameras": {k: v.to_dict() for k, v in self.cameras.items()},
            "flight_paths": {k: v.to_dict() for k, v in self.flight_paths.items()},
            "origin_lat": self.origin_lat,
            "origin_lon": self.origin_lon,
            "rotation": self.rotation,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GardenMap":
        """Deserialize from dictionary."""
        garden = cls()

        if data.get("boundary"):
            garden.boundary = Polygon.from_dict(data["boundary"])

        for zone_data in data.get("zones", {}).values():
            garden.add_zone(Zone.from_dict(zone_data))

        for cam_data in data.get("cameras", {}).values():
            garden.add_camera(CameraPlacement.from_dict(cam_data))

        for path_data in data.get("flight_paths", {}).values():
            garden.add_flight_path(FlightPath.from_dict(path_data))

        garden.origin_lat = data.get("origin_lat")
        garden.origin_lon = data.get("origin_lon")
        garden.rotation = data.get("rotation", 0.0)

        return garden

    def save(self, path: Path):
        """Save garden map to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Garden map saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "GardenMap":
        """Load garden map from JSON file."""
        with open(path) as f:
            data = json.load(f)
        logger.info(f"Garden map loaded from {path}")
        return cls.from_dict(data)


def create_example_garden() -> GardenMap:
    """Create an example garden map for testing."""
    garden = GardenMap()

    # Garden boundary (20m x 30m)
    garden.set_boundary(Polygon.rectangle(0, 0, 20, 30))

    # Chicken coop (protected zone)
    garden.add_zone(Zone(
        zone_id="coop",
        name="Chicken Coop",
        zone_type=ZoneType.PROTECTED,
        boundary=Polygon.rectangle(15, 20, 4, 5),
        priority=10,
        color="#ff0000",
    ))

    # Feeding area
    garden.add_zone(Zone(
        zone_id="feeding",
        name="Feeding Area",
        zone_type=ZoneType.FEEDING,
        boundary=Polygon.rectangle(8, 15, 6, 6),
        priority=5,
        color="#ffaa00",
    ))

    # Entry points
    garden.add_zone(Zone(
        zone_id="gate",
        name="Garden Gate",
        zone_type=ZoneType.ENTRY_POINT,
        boundary=Polygon.rectangle(0, 12, 1, 3),
        priority=0,
        alert_on_entry=True,
        color="#0000ff",
    ))

    garden.add_zone(Zone(
        zone_id="fence_gap",
        name="Fence Gap",
        zone_type=ZoneType.ENTRY_POINT,
        boundary=Polygon.rectangle(19, 5, 1, 2),
        priority=0,
        alert_on_entry=True,
        color="#0000ff",
    ))

    # Cameras
    garden.add_camera(CameraPlacement(
        camera_id="cam1",
        name="Coop Camera",
        position=Point(17, 25),
        heading=180,  # Looking south
        range=12,
    ))

    garden.add_camera(CameraPlacement(
        camera_id="cam2",
        name="Gate Camera",
        position=Point(2, 10),
        heading=90,  # Looking east
        range=15,
    ))

    garden.add_camera(CameraPlacement(
        camera_id="cam3",
        name="East Fence Camera",
        position=Point(18, 8),
        heading=270,  # Looking west
        range=15,
    ))

    return garden
