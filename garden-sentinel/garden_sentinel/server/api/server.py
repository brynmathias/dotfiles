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
_connected_websockets: list[WebSocket] = []


def create_app(
    detection_pipeline,
    alert_manager,
    storage_manager,
    mqtt_handler=None,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    global _detection_pipeline, _alert_manager, _storage_manager, _mqtt_handler

    _detection_pipeline = detection_pipeline
    _alert_manager = alert_manager
    _storage_manager = storage_manager
    _mqtt_handler = mqtt_handler

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
    ):
        """
        Upload a frame from an edge device for detection.
        """
        try:
            # Read frame data
            frame_bytes = await frame.read()

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
        }
        .header h1 { color: #00d9ff; display: flex; align-items: center; gap: 10px; }
        .container { padding: 20px; max-width: 1400px; margin: 0 auto; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .card h2 { color: #00d9ff; margin-bottom: 15px; font-size: 1.2rem; }
        .stat { margin: 10px 0; }
        .stat-label { color: #888; font-size: 0.9rem; }
        .stat-value { font-size: 1.5rem; font-weight: bold; color: #fff; }
        .alert-item {
            background: rgba(255, 0, 0, 0.1);
            border-left: 3px solid #ff4444;
            padding: 10px;
            margin: 10px 0;
            border-radius: 0 8px 8px 0;
        }
        .alert-critical { border-color: #ff0000; background: rgba(255, 0, 0, 0.2); }
        .alert-high { border-color: #ff8800; background: rgba(255, 136, 0, 0.2); }
        .alert-medium { border-color: #ffcc00; background: rgba(255, 204, 0, 0.2); }
        .device-status {
            display: flex; align-items: center; gap: 8px;
            padding: 8px 0; border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        .status-dot { width: 10px; height: 10px; border-radius: 50%; }
        .status-online { background: #00ff88; }
        .status-offline { background: #ff4444; }
        .btn {
            background: #00d9ff; color: #000; border: none;
            padding: 8px 16px; border-radius: 6px; cursor: pointer;
            font-weight: bold; transition: all 0.2s;
        }
        .btn:hover { background: #00b8d9; }
        .btn-danger { background: #ff4444; color: #fff; }
        .btn-danger:hover { background: #ff2222; }
        #connection-status {
            position: fixed; top: 10px; right: 10px;
            padding: 5px 10px; border-radius: 20px;
            font-size: 0.8rem;
        }
        .connected { background: rgba(0, 255, 136, 0.2); color: #00ff88; }
        .disconnected { background: rgba(255, 68, 68, 0.2); color: #ff4444; }
    </style>
</head>
<body>
    <div id="connection-status" class="disconnected">Disconnected</div>

    <header class="header">
        <h1>🐔 Garden Sentinel</h1>
        <p>Protecting your chickens from predators</p>
    </header>

    <div class="container">
        <div class="grid">
            <div class="card">
                <h2>📊 Detection Statistics</h2>
                <div class="stat">
                    <div class="stat-label">Frames Processed</div>
                    <div class="stat-value" id="frames-processed">0</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Total Detections</div>
                    <div class="stat-value" id="detections-count">0</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Alerts Triggered</div>
                    <div class="stat-value" id="alerts-triggered">0</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Avg Inference Time</div>
                    <div class="stat-value" id="avg-inference">0 ms</div>
                </div>
            </div>

            <div class="card">
                <h2>📹 Devices</h2>
                <div id="devices-list">
                    <p style="color: #888;">Loading devices...</p>
                </div>
            </div>

            <div class="card">
                <h2>🚨 Recent Alerts</h2>
                <div id="alerts-list">
                    <p style="color: #888;">No recent alerts</p>
                </div>
            </div>

            <div class="card">
                <h2>🎯 Quick Actions</h2>
                <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                    <button class="btn" onclick="testAlarm()">Test Alarm</button>
                    <button class="btn" onclick="testSprayer()">Test Sprayer</button>
                    <button class="btn btn-danger" onclick="stopAll()">Stop All</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let reconnectAttempts = 0;

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

                if (message.type === 'stats') {
                    updateStats(message.data);
                } else if (message.type === 'alert') {
                    addAlert(message.data);
                }
            };
        }

        function updateStats(stats) {
            if (stats.detection) {
                document.getElementById('frames-processed').textContent =
                    stats.detection.frames_processed.toLocaleString();
                document.getElementById('detections-count').textContent =
                    stats.detection.detections_count.toLocaleString();
                document.getElementById('alerts-triggered').textContent =
                    stats.detection.alerts_triggered.toLocaleString();
                document.getElementById('avg-inference').textContent =
                    stats.detection.avg_inference_time_ms.toFixed(1) + ' ms';
            }
        }

        function addAlert(alert) {
            const list = document.getElementById('alerts-list');
            const item = document.createElement('div');
            item.className = `alert-item alert-${alert.threat_level}`;
            item.innerHTML = `
                <strong>${alert.device_id}</strong> - ${alert.threat_level.toUpperCase()}
                <br><small>${new Date(alert.timestamp).toLocaleString()}</small>
            `;
            list.prepend(item);

            while (list.children.length > 10) {
                list.removeChild(list.lastChild);
            }
        }

        async function loadDevices() {
            try {
                const response = await fetch('/api/devices');
                const data = await response.json();
                const list = document.getElementById('devices-list');

                if (data.devices.length === 0) {
                    list.innerHTML = '<p style="color: #888;">No devices registered</p>';
                    return;
                }

                list.innerHTML = data.devices.map(device => `
                    <div class="device-status">
                        <span class="status-dot ${device.status === 'online' ? 'status-online' : 'status-offline'}"></span>
                        <span>${device.name || device.id}</span>
                        <small style="color: #888; margin-left: auto;">${device.location || 'Unknown'}</small>
                    </div>
                `).join('');
            } catch (e) {
                console.error('Failed to load devices:', e);
            }
        }

        async function sendCommand(deviceId, command, params = {}) {
            await fetch(`/api/devices/${deviceId}/command`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ device_id: deviceId, command, parameters: params })
            });
        }

        function testAlarm() { sendCommand('*', 'activate_alarm', { duration_s: 3 }); }
        function testSprayer() { sendCommand('*', 'activate_sprayer', { duration_s: 2 }); }
        function stopAll() {
            sendCommand('*', 'deactivate_alarm');
            sendCommand('*', 'deactivate_sprayer');
        }

        // Initialize
        connect();
        loadDevices();
        setInterval(loadDevices, 30000);
    </script>
</body>
</html>"""
