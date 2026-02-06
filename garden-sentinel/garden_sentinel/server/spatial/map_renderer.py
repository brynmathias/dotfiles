"""
Map visualization and rendering for Garden Sentinel.

Provides APIs to generate map visualizations showing:
- Garden boundaries and zones
- Camera positions and coverage
- Active predator tracks
- Entry points and hotspots
- Drone flight paths
"""

import json
import math
import time
from dataclasses import dataclass, asdict
from typing import Optional
from enum import Enum

from .garden_map import GardenMap, Point, Zone, CameraPlacement, FlightPath
from .zone_tracker import ZoneTracker, PredatorTrack


class RenderFormat(Enum):
    """Output format for map rendering."""
    JSON = "json"      # GeoJSON-compatible for web maps
    SVG = "svg"        # Vector graphics
    GEOJSON = "geojson"  # Standard GeoJSON


@dataclass
class RenderOptions:
    """Options for map rendering."""
    # View options
    width: int = 800
    height: int = 600
    padding: int = 20

    # Feature toggles
    show_zones: bool = True
    show_cameras: bool = True
    show_coverage: bool = True
    show_tracks: bool = True
    show_flight_paths: bool = True
    show_grid: bool = False

    # Style options
    zone_opacity: float = 0.3
    coverage_opacity: float = 0.15
    track_width: float = 2.0
    camera_size: float = 10.0

    # Color schemes
    zone_colors: Optional[dict[str, str]] = None
    predator_colors: Optional[dict[str, str]] = None

    # Time range for tracks (seconds, 0 = all)
    track_time_window: float = 300.0


# Default colors
DEFAULT_ZONE_COLORS = {
    "protected": "#4CAF50",    # Green
    "feeding": "#FFC107",      # Amber
    "perimeter": "#2196F3",    # Blue
    "entry_point": "#FF5722",  # Deep Orange
    "exclusion": "#F44336",    # Red
    "patrol": "#9C27B0",       # Purple
}

DEFAULT_PREDATOR_COLORS = {
    "fox": "#FF6B35",
    "badger": "#4A4A4A",
    "cat": "#8B4513",
    "bird_of_prey": "#4169E1",
    "rat": "#696969",
    "mink": "#8B008B",
    "unknown": "#808080",
}


class MapRenderer:
    """
    Renders garden maps with real-time tracking data.

    Supports multiple output formats for different use cases:
    - JSON/GeoJSON for web-based map visualization
    - SVG for static image generation
    """

    def __init__(
        self,
        garden_map: GardenMap,
        zone_tracker: Optional[ZoneTracker] = None,
    ):
        self.garden_map = garden_map
        self.zone_tracker = zone_tracker

    def render(
        self,
        format: RenderFormat = RenderFormat.JSON,
        options: Optional[RenderOptions] = None,
    ) -> str:
        """
        Render the map to the specified format.

        Returns string representation of the rendered map.
        """
        options = options or RenderOptions()

        if format == RenderFormat.JSON:
            return self._render_json(options)
        elif format == RenderFormat.GEOJSON:
            return self._render_geojson(options)
        elif format == RenderFormat.SVG:
            return self._render_svg(options)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def get_map_state(self, options: Optional[RenderOptions] = None) -> dict:
        """
        Get complete map state as a dictionary.

        Useful for web API endpoints.
        """
        options = options or RenderOptions()

        state = {
            "timestamp": time.time(),
            "bounds": self._get_bounds(),
            "zones": [],
            "cameras": [],
            "tracks": [],
            "flight_paths": [],
            "statistics": {},
        }

        # Add zones
        if options.show_zones:
            for zone in self.garden_map.zones:
                state["zones"].append(self._zone_to_dict(zone, options))

        # Add cameras
        if options.show_cameras:
            for camera in self.garden_map.cameras:
                state["cameras"].append(self._camera_to_dict(camera, options))

        # Add active tracks
        if options.show_tracks and self.zone_tracker:
            now = time.time()
            for track in self.zone_tracker.get_active_tracks():
                # Filter by time window
                if options.track_time_window > 0:
                    if now - track.last_seen > options.track_time_window:
                        continue
                state["tracks"].append(self._track_to_dict(track, options))

        # Add flight paths
        if options.show_flight_paths:
            for path in self.garden_map.flight_paths:
                state["flight_paths"].append(self._flight_path_to_dict(path))

        # Add statistics
        if self.zone_tracker:
            state["statistics"] = {
                "active_tracks": len(self.zone_tracker.get_active_tracks()),
                "zone_stats": self.zone_tracker.get_zone_statistics(),
                "entry_point_stats": self.zone_tracker.get_entry_point_statistics(),
            }

        return state

    def _get_bounds(self) -> dict:
        """Get map bounds."""
        if not self.garden_map.boundary:
            # Calculate from zones and cameras
            all_points = []
            for zone in self.garden_map.zones:
                all_points.extend(zone.polygon.vertices)
            for camera in self.garden_map.cameras:
                all_points.append(camera.position)

            if not all_points:
                return {"min_x": 0, "min_y": 0, "max_x": 100, "max_y": 100}

            return {
                "min_x": min(p.x for p in all_points),
                "min_y": min(p.y for p in all_points),
                "max_x": max(p.x for p in all_points),
                "max_y": max(p.y for p in all_points),
            }

        vertices = self.garden_map.boundary.vertices
        return {
            "min_x": min(p.x for p in vertices),
            "min_y": min(p.y for p in vertices),
            "max_x": max(p.x for p in vertices),
            "max_y": max(p.y for p in vertices),
        }

    def _zone_to_dict(self, zone: Zone, options: RenderOptions) -> dict:
        """Convert zone to dictionary."""
        colors = options.zone_colors or DEFAULT_ZONE_COLORS
        color = colors.get(zone.zone_type.value, "#808080")

        return {
            "id": zone.zone_id,
            "name": zone.name,
            "type": zone.zone_type.value,
            "color": color,
            "opacity": options.zone_opacity,
            "vertices": [[p.x, p.y] for p in zone.polygon.vertices],
            "centroid": [zone.polygon.centroid.x, zone.polygon.centroid.y],
            "area": zone.polygon.area,
        }

    def _camera_to_dict(self, camera: CameraPlacement, options: RenderOptions) -> dict:
        """Convert camera to dictionary."""
        result = {
            "id": camera.camera_id,
            "name": camera.name,
            "position": [camera.position.x, camera.position.y],
            "heading": camera.heading,
            "fov": camera.fov,
            "range": camera.range,
            "is_mobile": camera.is_mobile,
        }

        if camera.altitude is not None:
            result["altitude"] = camera.altitude

        # Add coverage cone if requested
        if options.show_coverage:
            result["coverage"] = self._get_coverage_polygon(camera)

        return result

    def _get_coverage_polygon(self, camera: CameraPlacement) -> list[list[float]]:
        """Generate coverage cone polygon for a camera."""
        # Camera position
        cx, cy = camera.position.x, camera.position.y

        # Calculate cone edges
        half_fov = camera.fov / 2
        left_angle = math.radians(camera.heading - half_fov)
        right_angle = math.radians(camera.heading + half_fov)

        # Generate arc points
        arc_points = []
        num_points = 20
        for i in range(num_points + 1):
            angle = left_angle + (right_angle - left_angle) * i / num_points
            x = cx + camera.range * math.sin(angle)
            y = cy + camera.range * math.cos(angle)
            arc_points.append([x, y])

        # Build polygon: camera position -> arc -> back to camera
        polygon = [[cx, cy]] + arc_points

        return polygon

    def _track_to_dict(self, track: PredatorTrack, options: RenderOptions) -> dict:
        """Convert predator track to dictionary."""
        colors = options.predator_colors or DEFAULT_PREDATOR_COLORS
        color = colors.get(track.predator_type, colors["unknown"])

        trajectory = [[p.x, p.y] for p in track.get_trajectory()]

        result = {
            "id": track.track_id,
            "predator_type": track.predator_type,
            "color": color,
            "trajectory": trajectory,
            "current_position": (
                [track.current_position.x, track.current_position.y]
                if track.current_position else None
            ),
            "first_seen": track.first_seen,
            "last_seen": track.last_seen,
            "total_distance": track.total_distance,
            "current_zones": list(track.current_zones),
            "was_sprayed": track.was_sprayed,
        }

        velocity = track.velocity
        if velocity:
            result["velocity"] = list(velocity)
            result["speed"] = math.sqrt(velocity[0]**2 + velocity[1]**2)

        return result

    def _flight_path_to_dict(self, path: FlightPath) -> dict:
        """Convert flight path to dictionary."""
        return {
            "id": path.path_id,
            "name": path.name,
            "waypoints": [[p.x, p.y] for p in path.waypoints],
            "altitudes": list(path.altitudes),
            "is_loop": path.is_loop,
        }

    def _render_json(self, options: RenderOptions) -> str:
        """Render to JSON format."""
        state = self.get_map_state(options)
        return json.dumps(state, indent=2)

    def _render_geojson(self, options: RenderOptions) -> str:
        """Render to GeoJSON format."""
        features = []

        # Add zones as polygons
        if options.show_zones:
            for zone in self.garden_map.zones:
                colors = options.zone_colors or DEFAULT_ZONE_COLORS
                color = colors.get(zone.zone_type.value, "#808080")

                # Close the polygon ring
                coords = [[p.x, p.y] for p in zone.polygon.vertices]
                coords.append(coords[0])

                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords],
                    },
                    "properties": {
                        "id": zone.zone_id,
                        "name": zone.name,
                        "type": zone.zone_type.value,
                        "color": color,
                        "featureType": "zone",
                    },
                })

        # Add cameras as points
        if options.show_cameras:
            for camera in self.garden_map.cameras:
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [camera.position.x, camera.position.y],
                    },
                    "properties": {
                        "id": camera.camera_id,
                        "name": camera.name,
                        "heading": camera.heading,
                        "fov": camera.fov,
                        "range": camera.range,
                        "featureType": "camera",
                    },
                })

                # Add coverage as separate polygon
                if options.show_coverage:
                    coverage = self._get_coverage_polygon(camera)
                    coverage.append(coverage[0])  # Close ring

                    features.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [coverage],
                        },
                        "properties": {
                            "camera_id": camera.camera_id,
                            "featureType": "coverage",
                        },
                    })

        # Add tracks as LineStrings
        if options.show_tracks and self.zone_tracker:
            colors = options.predator_colors or DEFAULT_PREDATOR_COLORS

            for track in self.zone_tracker.get_active_tracks():
                trajectory = track.get_trajectory()
                if len(trajectory) < 2:
                    continue

                color = colors.get(track.predator_type, colors["unknown"])

                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[p.x, p.y] for p in trajectory],
                    },
                    "properties": {
                        "track_id": track.track_id,
                        "predator_type": track.predator_type,
                        "color": color,
                        "featureType": "track",
                    },
                })

                # Add current position as point
                if track.current_position:
                    features.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [track.current_position.x, track.current_position.y],
                        },
                        "properties": {
                            "track_id": track.track_id,
                            "predator_type": track.predator_type,
                            "color": color,
                            "featureType": "predator_position",
                        },
                    })

        # Add flight paths as LineStrings
        if options.show_flight_paths:
            for path in self.garden_map.flight_paths:
                coords = [[p.x, p.y] for p in path.waypoints]
                if path.is_loop and coords:
                    coords.append(coords[0])

                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coords,
                    },
                    "properties": {
                        "path_id": path.path_id,
                        "name": path.name,
                        "featureType": "flight_path",
                    },
                })

        geojson = {
            "type": "FeatureCollection",
            "features": features,
        }

        return json.dumps(geojson, indent=2)

    def _render_svg(self, options: RenderOptions) -> str:
        """Render to SVG format."""
        bounds = self._get_bounds()

        # Calculate scaling
        map_width = bounds["max_x"] - bounds["min_x"]
        map_height = bounds["max_y"] - bounds["min_y"]

        if map_width == 0 or map_height == 0:
            map_width = map_height = 100

        available_width = options.width - 2 * options.padding
        available_height = options.height - 2 * options.padding

        scale = min(available_width / map_width, available_height / map_height)

        def transform_x(x: float) -> float:
            return options.padding + (x - bounds["min_x"]) * scale

        def transform_y(y: float) -> float:
            # Flip Y axis for SVG
            return options.height - options.padding - (y - bounds["min_y"]) * scale

        elements = []

        # SVG header
        elements.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{options.width}" height="{options.height}" '
            f'viewBox="0 0 {options.width} {options.height}">'
        )

        # Background
        elements.append(
            f'<rect width="{options.width}" height="{options.height}" fill="#f5f5f5"/>'
        )

        # Grid
        if options.show_grid:
            elements.append('<g class="grid" stroke="#ddd" stroke-width="0.5">')
            grid_size = 10  # meters
            for x in range(int(bounds["min_x"]), int(bounds["max_x"]) + 1, grid_size):
                tx = transform_x(x)
                elements.append(
                    f'<line x1="{tx}" y1="{options.padding}" '
                    f'x2="{tx}" y2="{options.height - options.padding}"/>'
                )
            for y in range(int(bounds["min_y"]), int(bounds["max_y"]) + 1, grid_size):
                ty = transform_y(y)
                elements.append(
                    f'<line x1="{options.padding}" y1="{ty}" '
                    f'x2="{options.width - options.padding}" y2="{ty}"/>'
                )
            elements.append('</g>')

        # Zones
        if options.show_zones:
            elements.append('<g class="zones">')
            colors = options.zone_colors or DEFAULT_ZONE_COLORS

            for zone in self.garden_map.zones:
                color = colors.get(zone.zone_type.value, "#808080")
                points = " ".join(
                    f"{transform_x(p.x)},{transform_y(p.y)}"
                    for p in zone.polygon.vertices
                )
                elements.append(
                    f'<polygon points="{points}" '
                    f'fill="{color}" fill-opacity="{options.zone_opacity}" '
                    f'stroke="{color}" stroke-width="1"/>'
                )

                # Zone label
                cx = transform_x(zone.polygon.centroid.x)
                cy = transform_y(zone.polygon.centroid.y)
                elements.append(
                    f'<text x="{cx}" y="{cy}" text-anchor="middle" '
                    f'font-size="10" fill="#333">{zone.name}</text>'
                )
            elements.append('</g>')

        # Camera coverage
        if options.show_coverage:
            elements.append('<g class="coverage">')
            for camera in self.garden_map.cameras:
                coverage = self._get_coverage_polygon(camera)
                points = " ".join(
                    f"{transform_x(p[0])},{transform_y(p[1])}"
                    for p in coverage
                )
                elements.append(
                    f'<polygon points="{points}" '
                    f'fill="#2196F3" fill-opacity="{options.coverage_opacity}" '
                    f'stroke="#2196F3" stroke-width="0.5" stroke-dasharray="4,2"/>'
                )
            elements.append('</g>')

        # Tracks
        if options.show_tracks and self.zone_tracker:
            elements.append('<g class="tracks">')
            colors = options.predator_colors or DEFAULT_PREDATOR_COLORS

            for track in self.zone_tracker.get_active_tracks():
                trajectory = track.get_trajectory()
                if len(trajectory) < 2:
                    continue

                color = colors.get(track.predator_type, colors["unknown"])
                points = " ".join(
                    f"{transform_x(p.x)},{transform_y(p.y)}"
                    for p in trajectory
                )
                elements.append(
                    f'<polyline points="{points}" '
                    f'fill="none" stroke="{color}" '
                    f'stroke-width="{options.track_width}"/>'
                )

                # Current position marker
                if track.current_position:
                    cx = transform_x(track.current_position.x)
                    cy = transform_y(track.current_position.y)
                    elements.append(
                        f'<circle cx="{cx}" cy="{cy}" r="5" '
                        f'fill="{color}" stroke="white" stroke-width="1"/>'
                    )
            elements.append('</g>')

        # Flight paths
        if options.show_flight_paths:
            elements.append('<g class="flight-paths">')
            for path in self.garden_map.flight_paths:
                if not path.waypoints:
                    continue

                coords = [[p.x, p.y] for p in path.waypoints]
                if path.is_loop:
                    coords.append(coords[0])

                points = " ".join(
                    f"{transform_x(c[0])},{transform_y(c[1])}"
                    for c in coords
                )
                elements.append(
                    f'<polyline points="{points}" '
                    f'fill="none" stroke="#9C27B0" '
                    f'stroke-width="2" stroke-dasharray="8,4"/>'
                )

                # Waypoint markers
                for wp in path.waypoints:
                    wx = transform_x(wp.x)
                    wy = transform_y(wp.y)
                    elements.append(
                        f'<circle cx="{wx}" cy="{wy}" r="3" '
                        f'fill="#9C27B0" stroke="white" stroke-width="1"/>'
                    )
            elements.append('</g>')

        # Cameras
        if options.show_cameras:
            elements.append('<g class="cameras">')
            for camera in self.garden_map.cameras:
                cx = transform_x(camera.position.x)
                cy = transform_y(camera.position.y)

                # Camera icon (triangle pointing in heading direction)
                angle = math.radians(-camera.heading)  # Negative for SVG coords
                size = options.camera_size

                # Triangle points
                p1 = (cx + size * math.sin(angle), cy - size * math.cos(angle))
                p2 = (
                    cx + size * 0.5 * math.sin(angle + 2.5),
                    cy - size * 0.5 * math.cos(angle + 2.5)
                )
                p3 = (
                    cx + size * 0.5 * math.sin(angle - 2.5),
                    cy - size * 0.5 * math.cos(angle - 2.5)
                )

                points = f"{p1[0]},{p1[1]} {p2[0]},{p2[1]} {p3[0]},{p3[1]}"

                fill_color = "#E91E63" if camera.is_mobile else "#2196F3"
                elements.append(
                    f'<polygon points="{points}" '
                    f'fill="{fill_color}" stroke="white" stroke-width="1"/>'
                )

                # Camera label
                elements.append(
                    f'<text x="{cx}" y="{cy + size + 10}" text-anchor="middle" '
                    f'font-size="8" fill="#333">{camera.name}</text>'
                )
            elements.append('</g>')

        # Close SVG
        elements.append('</svg>')

        return "\n".join(elements)


class MapAPIEndpoint:
    """
    Simple API endpoint handler for map data.

    Can be integrated with web frameworks like FastAPI or Flask.
    """

    def __init__(
        self,
        garden_map: GardenMap,
        zone_tracker: Optional[ZoneTracker] = None,
    ):
        self.renderer = MapRenderer(garden_map, zone_tracker)

    def get_state(self) -> dict:
        """Get complete map state for web clients."""
        return self.renderer.get_map_state()

    def get_zones(self) -> list[dict]:
        """Get all zones."""
        return self.renderer.get_map_state().get("zones", [])

    def get_cameras(self) -> list[dict]:
        """Get all cameras."""
        return self.renderer.get_map_state().get("cameras", [])

    def get_tracks(self) -> list[dict]:
        """Get active predator tracks."""
        return self.renderer.get_map_state().get("tracks", [])

    def get_statistics(self) -> dict:
        """Get tracking statistics."""
        return self.renderer.get_map_state().get("statistics", {})

    def render_geojson(self) -> str:
        """Render map as GeoJSON."""
        return self.renderer.render(RenderFormat.GEOJSON)

    def render_svg(self, width: int = 800, height: int = 600) -> str:
        """Render map as SVG."""
        options = RenderOptions(width=width, height=height)
        return self.renderer.render(RenderFormat.SVG, options)
