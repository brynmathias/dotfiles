"""
FastAPI server for Garden Sentinel.
Handles frame ingestion, API endpoints, and WebSocket connections.
"""

import asyncio
import base64
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from garden_sentinel.shared import CommandType, ServerCommand, ThreatLevel


class FrameUpload(BaseModel):
    device_id: str
    timestamp: float


class CommandRequest(BaseModel):
    device_id: str
    command: str
    parameters: dict = {}


class ConfigUpdate(BaseModel):
    device_id: str
    config: dict


# Global references (set during app creation)
_detection_pipeline = None
_alert_manager = None
_storage_manager = None
_mqtt_handler = None
_camera_coordinator = None
_pattern_analyzer = None
_health_aggregator = None
_map_api = None
_drone_tracker = None
_metrics_collector = None
_connected_websockets: list[WebSocket] = []

# Device stream registry: device_id -> {"url": stream_url, "last_frame": bytes, "last_update": timestamp}
_device_streams: dict[str, dict] = {}


def create_app(
    detection_pipeline,
    alert_manager,
    storage_manager,
    mqtt_handler=None,
    camera_coordinator=None,
    pattern_analyzer=None,
    health_aggregator=None,
    map_api=None,
    drone_tracker=None,
    metrics_collector=None,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    global _detection_pipeline, _alert_manager, _storage_manager, _mqtt_handler
    global _camera_coordinator, _pattern_analyzer, _health_aggregator
    global _map_api, _drone_tracker, _metrics_collector

    _detection_pipeline = detection_pipeline
    _alert_manager = alert_manager
    _storage_manager = storage_manager
    _mqtt_handler = mqtt_handler
    _camera_coordinator = camera_coordinator
    _pattern_analyzer = pattern_analyzer
    _health_aggregator = health_aggregator
    _map_api = map_api
    _drone_tracker = drone_tracker
    _metrics_collector = metrics_collector

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        logger.info("API server starting up")

        # Start background tasks
        asyncio.create_task(broadcast_stats_loop())
        asyncio.create_task(cleanup_loop())

        yield

        # Shutdown
        logger.info("API server shutting down")

    app = FastAPI(
        title="Garden Sentinel API",
        description="API for the Garden Sentinel security camera system",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    register_routes(app)

    return app


def register_routes(app: FastAPI):
    """Register all API routes."""

    # ==================== Frame Ingestion ====================

    @app.post("/api/frames")
    async def upload_frame(
        frame: UploadFile = File(...),
        device_id: str = Form(...),
        timestamp: float = Form(...),
        stream_url: Optional[str] = Form(None),
    ):
        """
        Upload a frame from an edge device for detection.
        """
        try:
            # Read frame data
            frame_bytes = await frame.read()

            # Cache frame for dashboard live view
            _device_streams[device_id] = {
                "last_frame": frame_bytes,
                "last_update": timestamp,
                "stream_url": stream_url,
            }

            # Submit to detection pipeline
            if _detection_pipeline:
                _detection_pipeline.submit_frame_bytes(device_id, frame_bytes, timestamp)

            # Update device last seen
            if _storage_manager:
                _storage_manager.update_device(device_id)

            return {"status": "ok", "device_id": device_id}

        except Exception as e:
            logger.error(f"Frame upload error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ==================== Live Video Streams ====================

    @app.get("/api/devices/{device_id}/stream")
    async def get_device_stream_info(device_id: str):
        """Get stream information for a device."""
        if device_id not in _device_streams:
            raise HTTPException(status_code=404, detail="Device not found or no stream available")

        stream_info = _device_streams[device_id]
        return {
            "device_id": device_id,
            "stream_url": stream_info.get("stream_url"),
            "last_update": stream_info.get("last_update"),
            "has_frame": stream_info.get("last_frame") is not None,
        }

    @app.get("/api/devices/{device_id}/snapshot")
    async def get_device_snapshot(device_id: str):
        """Get the latest frame from a device."""
        if device_id not in _device_streams:
            raise HTTPException(status_code=404, detail="Device not found")

        frame_bytes = _device_streams[device_id].get("last_frame")
        if not frame_bytes:
            raise HTTPException(status_code=404, detail="No frame available")

        return StreamingResponse(
            iter([frame_bytes]),
            media_type="image/jpeg",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/api/devices/{device_id}/mjpeg")
    async def get_device_mjpeg_stream(device_id: str):
        """
        Get MJPEG stream for a device (proxied from cached frames).
        This provides a live view using the frames uploaded by edge devices.
        """
        async def generate():
            while True:
                if device_id in _device_streams:
                    frame_bytes = _device_streams[device_id].get("last_frame")
                    if frame_bytes:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n"
                            + frame_bytes + b"\r\n"
                        )
                await asyncio.sleep(0.1)  # ~10 FPS

        return StreamingResponse(
            generate(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/api/streams")
    async def list_streams():
        """List all available device streams."""
        streams = []
        for device_id, info in _device_streams.items():
            streams.append({
                "device_id": device_id,
                "stream_url": info.get("stream_url"),
                "last_update": info.get("last_update"),
                "proxy_url": f"/api/devices/{device_id}/mjpeg",
                "snapshot_url": f"/api/devices/{device_id}/snapshot",
            })
        return {"streams": streams}

    @app.post("/api/devices/{device_id}/aim")
    async def aim_device(device_id: str, x: float, y: float):
        """
        Command a device to aim at a specific point.
        x, y are normalized coordinates (0-1).
        """
        if _mqtt_handler:
            command = ServerCommand(
                target_device=device_id,
                command_type=CommandType.UPDATE_CONFIG,
                parameters={
                    "action": "aim",
                    "x": x,
                    "y": y,
                },
            )
            _mqtt_handler.send_command(command)

        return {"status": "ok", "aimed_at": {"x": x, "y": y}}

    # ==================== Device Management ====================

    @app.get("/api/devices")
    async def get_devices():
        """Get all registered devices."""
        if not _storage_manager:
            return {"devices": []}

        devices = _storage_manager.get_devices()
        return {"devices": devices}

    @app.post("/api/devices/{device_id}/command")
    async def send_command(device_id: str, request: CommandRequest):
        """Send a command to a device."""
        try:
            command_type = CommandType(request.command)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid command: {request.command}")

        if _alert_manager:
            _alert_manager.send_manual_command(device_id, command_type, request.parameters)

        return {"status": "ok", "command": request.command}

    @app.post("/api/devices/{device_id}/config")
    async def update_device_config(device_id: str, request: ConfigUpdate):
        """Update device configuration."""
        command = ServerCommand(
            target_device=device_id,
            command_type=CommandType.UPDATE_CONFIG,
            parameters={"config": request.config},
        )

        if _mqtt_handler:
            _mqtt_handler.send_command(command)

        return {"status": "ok"}

    # ==================== Events & Alerts ====================

    @app.get("/api/events")
    async def get_events(
        device_id: Optional[str] = None,
        threat_level: Optional[str] = None,
        limit: int = 100,
    ):
        """Get alert events."""
        if not _storage_manager:
            return {"events": []}

        events = _storage_manager.get_events(
            device_id=device_id,
            threat_level=threat_level,
            limit=limit,
        )

        return {"events": events}

    @app.get("/api/events/{event_id}")
    async def get_event(event_id: str):
        """Get a specific event."""
        if not _storage_manager:
            raise HTTPException(status_code=404, detail="Event not found")

        events = _storage_manager.get_events(limit=1)
        for event in events:
            if event.get("id") == event_id:
                return event

        raise HTTPException(status_code=404, detail="Event not found")

    @app.get("/api/events/{event_id}/frame")
    async def get_event_frame(event_id: str):
        """Get the frame associated with an event."""
        frame_path = Path(_storage_manager.config.events_dir) / event_id / "frame.jpg"

        if not frame_path.exists():
            raise HTTPException(status_code=404, detail="Frame not found")

        return FileResponse(frame_path, media_type="image/jpeg")

    # ==================== Statistics ====================

    @app.get("/api/stats")
    async def get_stats():
        """Get system statistics."""
        stats = {}

        if _detection_pipeline:
            stats["detection"] = _detection_pipeline.get_stats()

        if _storage_manager:
            stats["storage"] = _storage_manager.get_storage_stats()

        return stats

    @app.get("/api/stats/detection")
    async def get_detection_stats():
        """Get detection pipeline statistics."""
        if not _detection_pipeline:
            return {}

        return _detection_pipeline.get_stats()

    # ==================== Training Data ====================

    @app.get("/api/training/samples")
    async def get_training_samples(
        class_name: Optional[str] = None,
        verified_only: bool = False,
    ):
        """Get training samples."""
        if not _storage_manager:
            return {"samples": []}

        samples = _storage_manager.get_training_samples(
            class_name=class_name,
            verified_only=verified_only,
        )

        return {"samples": samples}

    @app.post("/api/training/samples/{sample_id}/verify")
    async def verify_training_sample(sample_id: int, verified: bool = True):
        """Mark a training sample as verified."""
        if not _storage_manager:
            raise HTTPException(status_code=500, detail="Storage not available")

        _storage_manager.mark_sample_verified(sample_id, verified)
        return {"status": "ok"}

    @app.get("/api/training/samples/{sample_id}/image")
    async def get_training_sample_image(sample_id: int):
        """Get training sample image."""
        if not _storage_manager:
            raise HTTPException(status_code=500, detail="Storage not available")

        samples = _storage_manager.get_training_samples()
        for sample in samples:
            if sample.get("id") == sample_id:
                image_path = Path(sample["image_path"])
                if image_path.exists():
                    return FileResponse(image_path, media_type="image/jpeg")

        raise HTTPException(status_code=404, detail="Sample not found")

    # ==================== WebSocket ====================

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket for real-time updates."""
        await websocket.accept()
        _connected_websockets.append(websocket)

        try:
            while True:
                # Keep connection alive and handle incoming messages
                data = await websocket.receive_text()
                message = json.loads(data)

                # Handle ping/pong
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

        except WebSocketDisconnect:
            _connected_websockets.remove(websocket)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            if websocket in _connected_websockets:
                _connected_websockets.remove(websocket)

    # ==================== Dashboard ====================

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        """Serve the dashboard HTML."""
        return get_dashboard_html()

    # ==================== Health Check ====================

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "ok",
            "detection_pipeline": _detection_pipeline.is_running if _detection_pipeline else False,
            "timestamp": datetime.now().isoformat(),
        }

    # ==================== Multi-Camera Coordination ====================

    @app.get("/api/coordination/targets")
    async def get_active_targets():
        """Get all currently tracked targets across cameras."""
        if not _camera_coordinator:
            return {"targets": []}

        targets = _camera_coordinator.get_active_targets()
        return {
            "targets": [
                {
                    "target_id": t.target_id,
                    "predator_type": t.predator_type,
                    "threat_level": t.threat_level.value,
                    "camera_count": t.camera_count,
                    "assigned_camera": t.assigned_camera,
                    "times_sprayed": t.times_sprayed,
                    "age_seconds": t.age,
                    "velocity": {"x": t.velocity_x, "y": t.velocity_y},
                }
                for t in targets
            ]
        }

    @app.get("/api/coordination/cameras")
    async def get_camera_status():
        """Get status of all coordinated cameras."""
        if not _camera_coordinator:
            return {"cameras": {}}

        return {"cameras": _camera_coordinator.get_camera_status()}

    @app.get("/api/coordination/targets/{target_id}")
    async def get_target(target_id: str):
        """Get details of a specific target."""
        if not _camera_coordinator:
            raise HTTPException(status_code=404, detail="Coordinator not available")

        target = _camera_coordinator.get_target(target_id)
        if not target:
            raise HTTPException(status_code=404, detail="Target not found")

        return {
            "target_id": target.target_id,
            "predator_type": target.predator_type,
            "threat_level": target.threat_level.value,
            "camera_count": target.camera_count,
            "assigned_camera": target.assigned_camera,
            "times_sprayed": target.times_sprayed,
            "first_seen": target.first_seen,
            "last_seen": target.last_seen,
            "cameras_detecting": list(target.camera_detections.keys()),
        }

    # ==================== Pattern Analytics ====================

    @app.get("/api/analytics/patterns")
    async def get_patterns():
        """Get learned predator patterns."""
        if not _pattern_analyzer:
            return {"patterns": {}}

        patterns = _pattern_analyzer.get_all_patterns()
        return {
            "patterns": {
                predator_type: {
                    "total_visits": p.total_visits,
                    "peak_windows": [str(w) for w in p.peak_windows],
                    "avg_visit_duration": p.avg_visit_duration,
                    "spray_effectiveness": p.spray_effectiveness,
                    "recent_trend": p.recent_trend,
                    "last_visit": p.last_visit.isoformat() if p.last_visit else None,
                    "hourly_distribution": p.hourly_counts,
                    "daily_distribution": p.daily_counts,
                }
                for predator_type, p in patterns.items()
            }
        }

    @app.get("/api/analytics/patterns/{predator_type}")
    async def get_pattern(predator_type: str):
        """Get pattern for a specific predator type."""
        if not _pattern_analyzer:
            raise HTTPException(status_code=404, detail="Analytics not available")

        pattern = _pattern_analyzer.get_pattern(predator_type)
        if not pattern:
            raise HTTPException(status_code=404, detail="No pattern for this predator type")

        return {
            "predator_type": predator_type,
            "total_visits": pattern.total_visits,
            "peak_windows": [str(w) for w in pattern.peak_windows],
            "avg_visit_duration": pattern.avg_visit_duration,
            "spray_effectiveness": pattern.spray_effectiveness,
            "visits_sprayed": pattern.visits_sprayed,
            "visits_deterred": pattern.visits_deterred,
            "recent_trend": pattern.recent_trend,
            "last_visit": pattern.last_visit.isoformat() if pattern.last_visit else None,
            "hourly_distribution": pattern.hourly_counts,
            "daily_distribution": pattern.daily_counts,
            "location_hotspots": pattern.location_hotspots,
        }

    @app.get("/api/analytics/risk")
    async def get_current_risk():
        """Get current risk levels for all predator types."""
        if not _pattern_analyzer:
            return {"risk_levels": {}}

        return {"risk_levels": _pattern_analyzer.get_current_risk_levels()}

    @app.get("/api/analytics/statistics")
    async def get_analytics_statistics():
        """Get overall analytics statistics."""
        if not _pattern_analyzer:
            return {}

        return await _pattern_analyzer.get_statistics()

    @app.get("/api/analytics/daily")
    async def get_daily_breakdown(days: int = 7):
        """Get daily visit breakdown."""
        if not _pattern_analyzer:
            return {"days": []}

        breakdown = await _pattern_analyzer.get_daily_breakdown(days)
        return {"days": breakdown}

    # ==================== Health Monitoring ====================

    @app.post("/api/health/report")
    async def submit_health_report(report: dict):
        """Receive health report from an edge device."""
        if not _health_aggregator:
            return {"status": "ok", "message": "Health aggregator not configured"}

        await _health_aggregator.process_health_report(report)
        return {"status": "ok"}

    @app.get("/api/health/fleet")
    async def get_fleet_health():
        """Get aggregated health for all devices."""
        if not _health_aggregator:
            return {"error": "Health aggregator not configured"}

        return _health_aggregator.get_fleet_health().to_dict()

    @app.get("/api/health/devices")
    async def get_all_device_health():
        """Get health status for all devices."""
        if not _health_aggregator:
            return {"devices": []}

        devices = _health_aggregator.get_all_devices()
        return {"devices": [d.to_dict() for d in devices]}

    @app.get("/api/health/devices/{device_id}")
    async def get_device_health(device_id: str):
        """Get health status for a specific device."""
        if not _health_aggregator:
            raise HTTPException(status_code=404, detail="Health aggregator not configured")

        device = _health_aggregator.get_device_health(device_id)
        if not device:
            raise HTTPException(status_code=404, detail="Device not found")

        return device.to_dict()

    @app.get("/api/health/devices/{device_id}/history")
    async def get_device_health_history(device_id: str, hours: int = 24):
        """Get health history for a specific device."""
        if not _health_aggregator:
            return {"history": []}

        history = await _health_aggregator.get_device_history(device_id, hours)
        return {"device_id": device_id, "hours": hours, "history": history}


async def broadcast_stats_loop():
    """Broadcast statistics to connected WebSocket clients."""
    while True:
        await asyncio.sleep(5)

        if not _connected_websockets:
            continue

        try:
            stats = {}
            if _detection_pipeline:
                stats["detection"] = _detection_pipeline.get_stats()

            message = {"type": "stats", "data": stats}

            for ws in _connected_websockets.copy():
                try:
                    await ws.send_json(message)
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Broadcast error: {e}")


async def cleanup_loop():
    """Periodic cleanup of old data."""
    while True:
        await asyncio.sleep(3600)  # Run every hour

        if _storage_manager:
            try:
                _storage_manager.cleanup_old_data()
            except Exception as e:
                logger.error(f"Cleanup error: {e}")


async def broadcast_alert(alert_data: dict):
    """Broadcast an alert to all connected WebSocket clients."""
    message = {"type": "alert", "data": alert_data}

    for ws in _connected_websockets.copy():
        try:
            await ws.send_json(message)
        except Exception:
            pass


def get_dashboard_html() -> str:
    """Return the dashboard HTML."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Garden Sentinel - Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
        }
        .header {
            background: rgba(0, 217, 255, 0.1);
            padding: 20px;
            border-bottom: 1px solid rgba(0, 217, 255, 0.3);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 { color: #00d9ff; display: flex; align-items: center; gap: 10px; }
        .header-controls { display: flex; gap: 10px; }
        .container { padding: 20px; max-width: 1800px; margin: 0 auto; }

        /* Video grid */
        .video-section { margin-bottom: 20px; }
        .video-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 15px;
        }
        .video-card {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 12px;
            overflow: hidden;
            border: 2px solid rgba(0, 217, 255, 0.3);
            position: relative;
        }
        .video-card.selected { border-color: #00ff88; }
        .video-header {
            background: rgba(0, 0, 0, 0.5);
            padding: 10px 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .video-title { font-weight: bold; color: #00d9ff; }
        .video-status { font-size: 0.8rem; color: #888; }
        .video-container {
            position: relative;
            background: #000;
            aspect-ratio: 16/9;
            cursor: crosshair;
        }
        .video-container img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        .video-overlay {
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            pointer-events: none;
        }
        .aim-marker {
            position: absolute;
            width: 40px; height: 40px;
            border: 2px solid #00ff88;
            border-radius: 50%;
            transform: translate(-50%, -50%);
            pointer-events: none;
            display: none;
        }
        .aim-marker::before, .aim-marker::after {
            content: '';
            position: absolute;
            background: #00ff88;
        }
        .aim-marker::before {
            width: 2px; height: 100%;
            left: 50%; transform: translateX(-50%);
        }
        .aim-marker::after {
            height: 2px; width: 100%;
            top: 50%; transform: translateY(-50%);
        }
        .video-controls {
            padding: 10px;
            display: flex;
            gap: 8px;
            background: rgba(0, 0, 0, 0.3);
        }
        .no-video {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #666;
            font-size: 1.2rem;
        }

        /* Stats grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 15px;
        }
        .card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .card h2 { color: #00d9ff; margin-bottom: 15px; font-size: 1.1rem; }
        .stat { margin: 8px 0; display: flex; justify-content: space-between; }
        .stat-label { color: #888; font-size: 0.9rem; }
        .stat-value { font-weight: bold; color: #fff; }
        .alert-item {
            background: rgba(255, 0, 0, 0.1);
            border-left: 3px solid #ff4444;
            padding: 10px;
            margin: 8px 0;
            border-radius: 0 8px 8px 0;
            font-size: 0.9rem;
        }
        .alert-critical { border-color: #ff0000; background: rgba(255, 0, 0, 0.2); }
        .alert-high { border-color: #ff8800; background: rgba(255, 136, 0, 0.2); }
        .alert-medium { border-color: #ffcc00; background: rgba(255, 204, 0, 0.1); }
        .btn {
            background: #00d9ff; color: #000; border: none;
            padding: 8px 16px; border-radius: 6px; cursor: pointer;
            font-weight: bold; transition: all 0.2s; font-size: 0.85rem;
        }
        .btn:hover { background: #00b8d9; }
        .btn-sm { padding: 5px 10px; font-size: 0.75rem; }
        .btn-danger { background: #ff4444; color: #fff; }
        .btn-danger:hover { background: #ff2222; }
        .btn-success { background: #00ff88; color: #000; }
        #connection-status {
            position: fixed; top: 10px; right: 10px;
            padding: 5px 10px; border-radius: 20px;
            font-size: 0.8rem; z-index: 100;
        }
        .connected { background: rgba(0, 255, 136, 0.2); color: #00ff88; }
        .disconnected { background: rgba(255, 68, 68, 0.2); color: #ff4444; }
        .targeting-status {
            position: absolute;
            bottom: 10px; left: 10px;
            background: rgba(0, 0, 0, 0.7);
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 0.8rem;
        }
        .targeting-status.engaged { color: #ff4444; }
        .targeting-status.tracking { color: #ffcc00; }
        .targeting-status.idle { color: #888; }
    </style>
</head>
<body>
    <div id="connection-status" class="disconnected">Disconnected</div>

    <header class="header">
        <div>
            <h1>Garden Sentinel</h1>
            <p style="color: #888; font-size: 0.9rem;">Protecting your chickens from predators</p>
        </div>
        <div class="header-controls">
            <button class="btn" onclick="testAlarm()">Test Alarm</button>
            <button class="btn" onclick="testSprayer()">Test Sprayer</button>
            <button class="btn btn-danger" onclick="stopAll()">Stop All</button>
        </div>
    </header>

    <div class="container">
        <!-- Live Video Feeds -->
        <section class="video-section">
            <h2 style="color: #00d9ff; margin-bottom: 15px;">Live Camera Feeds</h2>
            <p style="color: #666; margin-bottom: 15px; font-size: 0.9rem;">
                Click on video to aim camera/sprayer at that point
            </p>
            <div class="video-grid" id="video-grid">
                <div class="video-card">
                    <div class="video-container">
                        <div class="no-video">No cameras connected</div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Stats and Alerts -->
        <section class="stats-grid">
            <div class="card">
                <h2>Detection Statistics</h2>
                <div class="stat">
                    <span class="stat-label">Frames Processed</span>
                    <span class="stat-value" id="frames-processed">0</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Total Detections</span>
                    <span class="stat-value" id="detections-count">0</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Alerts Triggered</span>
                    <span class="stat-value" id="alerts-triggered">0</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Avg Inference Time</span>
                    <span class="stat-value" id="avg-inference">0 ms</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Queue Size</span>
                    <span class="stat-value" id="queue-size">0</span>
                </div>
            </div>

            <div class="card">
                <h2>Recent Alerts</h2>
                <div id="alerts-list">
                    <p style="color: #666; font-size: 0.9rem;">No recent alerts</p>
                </div>
            </div>

            <div class="card">
                <h2>Registered Devices</h2>
                <div id="devices-list">
                    <p style="color: #666; font-size: 0.9rem;">Loading...</p>
                </div>
            </div>
        </section>
    </div>

    <script>
        let ws = null;
        let reconnectAttempts = 0;
        let selectedDevice = null;

        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

            ws.onopen = () => {
                document.getElementById('connection-status').textContent = 'Connected';
                document.getElementById('connection-status').className = 'connected';
                reconnectAttempts = 0;
            };

            ws.onclose = () => {
                document.getElementById('connection-status').textContent = 'Disconnected';
                document.getElementById('connection-status').className = 'disconnected';
                setTimeout(connect, Math.min(1000 * Math.pow(2, reconnectAttempts++), 30000));
            };

            ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                if (message.type === 'stats') updateStats(message.data);
                else if (message.type === 'alert') addAlert(message.data);
                else if (message.type === 'detection') updateDetection(message.data);
            };
        }

        function updateStats(stats) {
            if (stats.detection) {
                document.getElementById('frames-processed').textContent =
                    stats.detection.frames_processed?.toLocaleString() || '0';
                document.getElementById('detections-count').textContent =
                    stats.detection.detections_count?.toLocaleString() || '0';
                document.getElementById('alerts-triggered').textContent =
                    stats.detection.alerts_triggered?.toLocaleString() || '0';
                document.getElementById('avg-inference').textContent =
                    (stats.detection.avg_inference_time_ms?.toFixed(1) || '0') + ' ms';
                document.getElementById('queue-size').textContent =
                    stats.detection.queue_size?.toString() || '0';
            }
        }

        function addAlert(alert) {
            const list = document.getElementById('alerts-list');
            // Remove "no alerts" message if present
            if (list.querySelector('p')) list.innerHTML = '';

            const item = document.createElement('div');
            item.className = `alert-item alert-${alert.threat_level}`;
            item.innerHTML = `
                <strong>${alert.device_id}</strong> - ${alert.threat_level?.toUpperCase() || 'ALERT'}
                <br><small>${new Date(alert.timestamp).toLocaleString()}</small>
            `;
            list.prepend(item);
            while (list.children.length > 8) list.removeChild(list.lastChild);

            // Flash the corresponding video card
            const card = document.querySelector(`[data-device="${alert.device_id}"]`);
            if (card) {
                card.style.borderColor = '#ff4444';
                setTimeout(() => card.style.borderColor = '', 2000);
            }
        }

        async function loadStreams() {
            try {
                const response = await fetch('/api/streams');
                const data = await response.json();
                const grid = document.getElementById('video-grid');

                if (!data.streams || data.streams.length === 0) {
                    grid.innerHTML = `
                        <div class="video-card">
                            <div class="video-container">
                                <div class="no-video">No cameras connected<br>
                                <small style="color:#555">Start an edge device to see video</small></div>
                            </div>
                        </div>`;
                    return;
                }

                grid.innerHTML = data.streams.map(stream => `
                    <div class="video-card" data-device="${stream.device_id}">
                        <div class="video-header">
                            <span class="video-title">${stream.device_id}</span>
                            <span class="video-status">Live</span>
                        </div>
                        <div class="video-container" onclick="handleVideoClick(event, '${stream.device_id}')">
                            <img src="${stream.proxy_url}" alt="Live feed" onerror="this.style.display='none'">
                            <div class="aim-marker" id="aim-${stream.device_id}"></div>
                            <div class="targeting-status idle" id="status-${stream.device_id}">IDLE</div>
                        </div>
                        <div class="video-controls">
                            <button class="btn btn-sm" onclick="centerCamera('${stream.device_id}')">Center</button>
                            <button class="btn btn-sm" onclick="testSprayerDevice('${stream.device_id}')">Spray</button>
                            <button class="btn btn-sm btn-danger" onclick="stopDevice('${stream.device_id}')">Stop</button>
                        </div>
                    </div>
                `).join('');

            } catch (e) {
                console.error('Failed to load streams:', e);
            }
        }

        async function loadDevices() {
            try {
                const response = await fetch('/api/devices');
                const data = await response.json();
                const list = document.getElementById('devices-list');

                if (!data.devices || data.devices.length === 0) {
                    list.innerHTML = '<p style="color: #666; font-size: 0.9rem;">No devices registered</p>';
                    return;
                }

                list.innerHTML = data.devices.map(device => `
                    <div style="display:flex;align-items:center;gap:8px;padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.1);">
                        <span style="width:8px;height:8px;border-radius:50%;background:${device.status === 'online' ? '#00ff88' : '#ff4444'}"></span>
                        <span>${device.name || device.id}</span>
                        <small style="color:#666;margin-left:auto">${device.location || ''}</small>
                    </div>
                `).join('');
            } catch (e) {
                console.error('Failed to load devices:', e);
            }
        }

        async function handleVideoClick(event, deviceId) {
            const container = event.currentTarget;
            const rect = container.getBoundingClientRect();
            const x = (event.clientX - rect.left) / rect.width;
            const y = (event.clientY - rect.top) / rect.height;

            // Show aim marker
            const marker = document.getElementById(`aim-${deviceId}`);
            if (marker) {
                marker.style.display = 'block';
                marker.style.left = `${x * 100}%`;
                marker.style.top = `${y * 100}%`;
                setTimeout(() => marker.style.display = 'none', 2000);
            }

            // Send aim command
            try {
                await fetch(`/api/devices/${deviceId}/aim?x=${x}&y=${y}`, { method: 'POST' });
                console.log(`Aimed ${deviceId} at (${x.toFixed(2)}, ${y.toFixed(2)})`);
            } catch (e) {
                console.error('Failed to aim:', e);
            }
        }

        async function sendCommand(deviceId, command, params = {}) {
            await fetch(`/api/devices/${deviceId}/command`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ device_id: deviceId, command, parameters: params })
            });
        }

        function centerCamera(deviceId) { handleVideoClick({ currentTarget: document.querySelector(`[data-device="${deviceId}"] .video-container`), clientX: 0, clientY: 0 }, deviceId); }
        function testSprayerDevice(deviceId) { sendCommand(deviceId, 'activate_sprayer', { duration_s: 2 }); }
        function stopDevice(deviceId) { sendCommand(deviceId, 'deactivate_sprayer'); sendCommand(deviceId, 'deactivate_alarm'); }
        function testAlarm() { sendCommand('*', 'activate_alarm', { duration_s: 3 }); }
        function testSprayer() { sendCommand('*', 'activate_sprayer', { duration_s: 2 }); }
        function stopAll() { sendCommand('*', 'deactivate_alarm'); sendCommand('*', 'deactivate_sprayer'); }

        // Initialize
        connect();
        loadStreams();
        loadDevices();
        setInterval(loadStreams, 10000);
        setInterval(loadDevices, 30000);
    </script>
</body>
</html>"""
