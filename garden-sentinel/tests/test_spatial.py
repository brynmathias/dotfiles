"""
Tests for the spatial module (geometry, zones, tracking).
"""

import pytest
import math
from garden_sentinel.server.spatial.garden_map import (
    Point,
    Polygon,
    Zone,
    ZoneType,
    CameraPlacement,
    FlightPath,
    GardenMap,
)
from garden_sentinel.server.spatial.zone_tracker import (
    ZoneTracker,
    ZoneEvent,
    ZoneEventType,
    PredatorTrack,
    DeterrenceTracker,
)
from garden_sentinel.server.spatial.drone_tracker import (
    GPSCoordinate,
    GPSConverter,
    DroneTracker,
    DroneStatus,
)


class TestPoint:
    """Tests for Point class."""

    def test_point_creation(self):
        p = Point(10.5, 20.3)
        assert p.x == 10.5
        assert p.y == 20.3

    def test_point_distance_to(self):
        p1 = Point(0, 0)
        p2 = Point(3, 4)
        assert p1.distance_to(p2) == 5.0

    def test_point_distance_to_same(self):
        p = Point(5, 5)
        assert p.distance_to(p) == 0.0


class TestPolygon:
    """Tests for Polygon class."""

    def test_polygon_creation(self):
        vertices = [Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10)]
        poly = Polygon(vertices)
        assert len(poly.vertices) == 4

    def test_polygon_area_square(self):
        vertices = [Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10)]
        poly = Polygon(vertices)
        assert poly.area == pytest.approx(100.0)

    def test_polygon_area_triangle(self):
        vertices = [Point(0, 0), Point(10, 0), Point(5, 10)]
        poly = Polygon(vertices)
        assert poly.area == pytest.approx(50.0)

    def test_polygon_centroid_square(self):
        vertices = [Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10)]
        poly = Polygon(vertices)
        centroid = poly.centroid
        assert centroid.x == pytest.approx(5.0)
        assert centroid.y == pytest.approx(5.0)

    def test_polygon_contains_point_inside(self):
        vertices = [Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10)]
        poly = Polygon(vertices)
        assert poly.contains_point(Point(5, 5)) is True

    def test_polygon_contains_point_outside(self):
        vertices = [Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10)]
        poly = Polygon(vertices)
        assert poly.contains_point(Point(15, 5)) is False

    def test_polygon_contains_point_on_edge(self):
        vertices = [Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10)]
        poly = Polygon(vertices)
        # Points on edge should be considered inside
        assert poly.contains_point(Point(5, 0)) is True


class TestZone:
    """Tests for Zone class."""

    def test_zone_creation(self):
        vertices = [Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10)]
        zone = Zone(
            zone_id="test_zone",
            name="Test Zone",
            zone_type=ZoneType.PROTECTED,
            polygon=Polygon(vertices),
        )
        assert zone.zone_id == "test_zone"
        assert zone.name == "Test Zone"
        assert zone.zone_type == ZoneType.PROTECTED

    def test_zone_types(self):
        assert ZoneType.PROTECTED.value == "protected"
        assert ZoneType.FEEDING.value == "feeding"
        assert ZoneType.PERIMETER.value == "perimeter"
        assert ZoneType.ENTRY_POINT.value == "entry_point"


class TestCameraPlacement:
    """Tests for CameraPlacement class."""

    def test_camera_creation(self):
        camera = CameraPlacement(
            camera_id="cam-01",
            name="Test Camera",
            position=Point(50, 0),
            heading=90,
            fov=60,
            range=20,
        )
        assert camera.camera_id == "cam-01"
        assert camera.heading == 90
        assert camera.fov == 60

    def test_camera_coverage_cone(self):
        camera = CameraPlacement(
            camera_id="cam-01",
            name="Test Camera",
            position=Point(0, 0),
            heading=0,  # North
            fov=90,
            range=10,
        )
        cone = camera.get_coverage_cone()
        # Should have camera position + arc points
        assert len(cone) > 1
        # First point should be camera position
        assert cone[0].x == 0
        assert cone[0].y == 0


class TestFlightPath:
    """Tests for FlightPath class."""

    def test_flight_path_creation(self):
        waypoints = [Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10)]
        path = FlightPath(
            path_id="patrol-1",
            name="Perimeter Patrol",
            waypoints=waypoints,
            altitudes=[15, 15, 15, 15],
            is_loop=True,
        )
        assert path.path_id == "patrol-1"
        assert len(path.waypoints) == 4
        assert path.is_loop is True


class TestGardenMap:
    """Tests for GardenMap class."""

    def test_garden_map_creation(self, sample_garden_config):
        # Create zones
        zones = []
        for zone_cfg in sample_garden_config["zones"]:
            vertices = [Point(p[0], p[1]) for p in zone_cfg["vertices"]]
            zone = Zone(
                zone_id=zone_cfg["id"],
                name=zone_cfg["name"],
                zone_type=ZoneType(zone_cfg["type"]),
                polygon=Polygon(vertices),
            )
            zones.append(zone)

        # Create cameras
        cameras = []
        for cam_cfg in sample_garden_config["cameras"]:
            camera = CameraPlacement(
                camera_id=cam_cfg["id"],
                name=cam_cfg["name"],
                position=Point(cam_cfg["x"], cam_cfg["y"]),
                heading=cam_cfg["heading"],
                fov=cam_cfg["fov"],
                range=cam_cfg["range"],
            )
            cameras.append(camera)

        garden_map = GardenMap(
            name=sample_garden_config["name"],
            zones=zones,
            cameras=cameras,
        )

        assert garden_map.name == "Test Garden"
        assert len(garden_map.zones) == 2
        assert len(garden_map.cameras) == 1


class TestZoneTracker:
    """Tests for ZoneTracker class."""

    @pytest.fixture
    def zone_tracker(self):
        # Create a simple garden map with one zone
        vertices = [Point(0, 0), Point(100, 0), Point(100, 100), Point(0, 100)]
        zone = Zone(
            zone_id="protected",
            name="Protected Area",
            zone_type=ZoneType.PROTECTED,
            polygon=Polygon(vertices),
        )
        garden_map = GardenMap(name="Test", zones=[zone])
        return ZoneTracker(garden_map=garden_map, track_timeout=60.0)

    def test_zone_tracker_creation(self, zone_tracker):
        assert zone_tracker is not None
        assert len(zone_tracker.active_tracks) == 0

    def test_update_position_new_track(self, zone_tracker):
        events = zone_tracker.update_position(
            track_id="fox_001",
            predator_type="fox",
            position=Point(50, 50),
            timestamp=1000.0,
        )

        assert "fox_001" in zone_tracker.active_tracks
        track = zone_tracker.active_tracks["fox_001"]
        assert track.predator_type == "fox"
        # Should have entered the protected zone
        assert len(events) == 1
        assert events[0].event_type == ZoneEventType.ENTERED

    def test_update_position_exit_zone(self, zone_tracker):
        # Enter zone
        zone_tracker.update_position(
            track_id="fox_001",
            predator_type="fox",
            position=Point(50, 50),
            timestamp=1000.0,
        )

        # Exit zone
        events = zone_tracker.update_position(
            track_id="fox_001",
            predator_type="fox",
            position=Point(150, 50),  # Outside zone
            timestamp=1005.0,
        )

        assert len(events) == 1
        assert events[0].event_type == ZoneEventType.EXITED
        assert events[0].duration_in_zone == pytest.approx(5.0)

    def test_mark_sprayed(self, zone_tracker):
        zone_tracker.update_position(
            track_id="fox_001",
            predator_type="fox",
            position=Point(50, 50),
            timestamp=1000.0,
        )

        zone_tracker.mark_sprayed("fox_001", timestamp=1002.0)

        track = zone_tracker.get_track("fox_001")
        assert track.was_sprayed is True
        assert track.spray_time == 1002.0

    def test_get_tracks_in_zone(self, zone_tracker):
        # Add two tracks in zone
        zone_tracker.update_position("fox_001", "fox", Point(30, 30), 1000.0)
        zone_tracker.update_position("cat_001", "cat", Point(70, 70), 1000.0)

        tracks = zone_tracker.get_tracks_in_zone("protected")
        assert len(tracks) == 2


class TestPredatorTrack:
    """Tests for PredatorTrack class."""

    def test_track_creation(self):
        track = PredatorTrack(
            track_id="fox_001",
            predator_type="fox",
            first_seen=1000.0,
            last_seen=1000.0,
        )
        assert track.track_id == "fox_001"
        assert track.total_distance == 0.0

    def test_track_add_position(self):
        track = PredatorTrack(
            track_id="fox_001",
            predator_type="fox",
            first_seen=1000.0,
            last_seen=1000.0,
        )

        track.add_position(Point(0, 0), 1000.0)
        track.add_position(Point(3, 4), 1001.0)  # 5 meters away

        assert track.total_distance == pytest.approx(5.0)
        assert track.current_position.x == 3
        assert track.current_position.y == 4

    def test_track_velocity(self):
        track = PredatorTrack(
            track_id="fox_001",
            predator_type="fox",
            first_seen=1000.0,
            last_seen=1000.0,
        )

        track.add_position(Point(0, 0), 1000.0)
        track.add_position(Point(10, 0), 1002.0)  # 10m in 2s = 5m/s

        velocity = track.velocity
        assert velocity is not None
        assert velocity[0] == pytest.approx(5.0)  # vx
        assert velocity[1] == pytest.approx(0.0)  # vy


class TestDeterrenceTracker:
    """Tests for DeterrenceTracker class."""

    @pytest.fixture
    def deterrence_setup(self):
        vertices = [Point(0, 0), Point(100, 0), Point(100, 100), Point(0, 100)]
        zone = Zone(
            zone_id="protected",
            name="Protected",
            zone_type=ZoneType.PROTECTED,
            polygon=Polygon(vertices),
        )
        garden_map = GardenMap(name="Test", zones=[zone])
        zone_tracker = ZoneTracker(garden_map=garden_map)
        deterrence_tracker = DeterrenceTracker(
            zone_tracker=zone_tracker,
            protected_zone_ids=["protected"],
            deterrence_window=30.0,
        )
        return zone_tracker, deterrence_tracker

    def test_record_spray(self, deterrence_setup):
        zone_tracker, deterrence_tracker = deterrence_setup

        # Enter zone
        zone_tracker.update_position("fox_001", "fox", Point(50, 50), 1000.0)

        # Record spray
        deterrence_tracker.record_spray("fox_001", timestamp=1005.0)

        assert len(deterrence_tracker.spray_events) == 1
        assert deterrence_tracker.spray_events[0]["track_id"] == "fox_001"

    def test_effectiveness_calculation(self, deterrence_setup):
        zone_tracker, deterrence_tracker = deterrence_setup

        # Enter zone
        zone_tracker.update_position("fox_001", "fox", Point(50, 50), 1000.0)

        # Spray
        deterrence_tracker.record_spray("fox_001", timestamp=1005.0)

        # Exit zone within 30s (deterred)
        zone_tracker.update_position("fox_001", "fox", Point(150, 50), 1020.0)

        effectiveness = deterrence_tracker.get_effectiveness()
        assert effectiveness["total_sprays"] == 1
        assert effectiveness["deterred_count"] == 1
        assert effectiveness["effectiveness"] == pytest.approx(1.0)


class TestGPSConverter:
    """Tests for GPS coordinate conversion."""

    def test_gps_converter_creation(self):
        origin = GPSCoordinate(latitude=51.5074, longitude=-0.1278, altitude=0)
        converter = GPSConverter(origin)
        assert converter.origin == origin

    def test_gps_to_local_same_point(self):
        origin = GPSCoordinate(latitude=51.5074, longitude=-0.1278, altitude=0)
        converter = GPSConverter(origin)

        local = converter.to_local(origin)
        assert local[0] == pytest.approx(0.0, abs=0.01)
        assert local[1] == pytest.approx(0.0, abs=0.01)
        assert local[2] == pytest.approx(0.0, abs=0.01)

    def test_gps_to_local_offset(self):
        origin = GPSCoordinate(latitude=51.5074, longitude=-0.1278, altitude=0)
        converter = GPSConverter(origin)

        # Move slightly north (larger latitude = more north = +y)
        point = GPSCoordinate(latitude=51.5084, longitude=-0.1278, altitude=0)
        local = converter.to_local(point)

        # Should be ~111m north (1 degree lat ≈ 111km)
        assert local[0] == pytest.approx(0.0, abs=1.0)
        assert local[1] > 0  # Positive y (north)
        assert abs(local[1]) == pytest.approx(111.0, rel=0.1)

    def test_local_to_gps_roundtrip(self):
        origin = GPSCoordinate(latitude=51.5074, longitude=-0.1278, altitude=0)
        converter = GPSConverter(origin)

        # Convert to local and back
        local = (100.0, 200.0, 50.0)
        gps = converter.to_gps(*local)
        back = converter.to_local(gps)

        assert back[0] == pytest.approx(local[0], abs=0.1)
        assert back[1] == pytest.approx(local[1], abs=0.1)
        assert back[2] == pytest.approx(local[2], abs=0.1)


class TestDroneTracker:
    """Tests for DroneTracker class."""

    def test_drone_tracker_creation(self):
        tracker = DroneTracker()
        assert len(tracker.drones) == 0

    def test_register_drone(self):
        tracker = DroneTracker()
        drone = tracker.register_drone("drone-01", camera_type="drone")

        assert drone.camera_id == "drone-01"
        assert drone.status == DroneStatus.OFFLINE
        assert "drone-01" in tracker.drones

    def test_update_position(self):
        origin = GPSCoordinate(latitude=51.5074, longitude=-0.1278, altitude=0)
        converter = GPSConverter(origin)
        tracker = DroneTracker(gps_converter=converter)

        tracker.register_drone("drone-01")

        gps = GPSCoordinate(latitude=51.5084, longitude=-0.1268, altitude=50)
        tracker.update_position(
            camera_id="drone-01",
            gps=gps,
            heading=90,
            speed=5.0,
            battery_percent=80.0,
        )

        drone = tracker.get_drone("drone-01")
        assert drone is not None
        assert drone.current_telemetry is not None
        assert drone.current_telemetry.heading == 90
        assert drone.current_telemetry.battery_percent == 80.0
        assert drone.current_telemetry.local_position is not None

    def test_get_active_drones(self):
        tracker = DroneTracker()
        tracker.register_drone("drone-01")
        tracker.register_drone("drone-02")

        # Update one with movement (makes it "flying")
        tracker.update_position(
            camera_id="drone-01",
            gps=GPSCoordinate(51.5, -0.1, 50),
            speed=10.0,
        )

        active = tracker.get_active_drones()
        assert len(active) == 1
        assert active[0].camera_id == "drone-01"
