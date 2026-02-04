"""
HTTP MJPEG streaming server for Garden Sentinel edge device.
Provides live video feed accessible via web browser.
"""

import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .camera import Camera

logger = logging.getLogger(__name__)


class StreamingHandler(BaseHTTPRequestHandler):
    """HTTP request handler for MJPEG streaming."""

    camera: "Camera" = None
    jpeg_quality: int = 85
    stream_fps: int = 15
    stream_path: str = "/stream"

    def log_message(self, format, *args):
        """Override to use our logger."""
        logger.debug(f"{self.address_string()} - {format % args}")

    def do_GET(self):
        """Handle GET requests."""
        if self.path == self.stream_path:
            self._handle_stream()
        elif self.path == "/snapshot":
            self._handle_snapshot()
        elif self.path == "/health":
            self._handle_health()
        elif self.path == "/":
            self._handle_index()
        else:
            self.send_error(404, "Not Found")

    def _handle_stream(self):
        """Handle MJPEG stream request."""
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        frame_interval = 1.0 / self.stream_fps

        try:
            while True:
                start_time = time.time()

                jpeg = self.camera.get_jpeg(quality=self.jpeg_quality)
                if jpeg is None:
                    time.sleep(0.1)
                    continue

                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    logger.debug("Client disconnected from stream")
                    break

                # Rate limiting
                elapsed = time.time() - start_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)

        except Exception as e:
            logger.error(f"Stream error: {e}")

    def _handle_snapshot(self):
        """Handle snapshot request."""
        jpeg = self.camera.get_jpeg(quality=95)
        if jpeg is None:
            self.send_error(503, "Camera not ready")
            return

        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(jpeg)

    def _handle_health(self):
        """Handle health check request."""
        response = b'{"status": "ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def _handle_index(self):
        """Handle index page with embedded stream."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Garden Sentinel - Live Feed</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background: #1a1a2e;
            color: #eee;
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        h1 {{ color: #00d9ff; }}
        .stream-container {{
            border: 3px solid #00d9ff;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 0 20px rgba(0, 217, 255, 0.3);
        }}
        img {{
            display: block;
            max-width: 100%;
            height: auto;
        }}
        .info {{
            margin-top: 20px;
            padding: 15px;
            background: #16213e;
            border-radius: 8px;
        }}
    </style>
</head>
<body>
    <h1>🐔 Garden Sentinel</h1>
    <div class="stream-container">
        <img src="{self.stream_path}" alt="Live Feed">
    </div>
    <div class="info">
        <p>Live camera feed from the garden security system.</p>
        <p><a href="/snapshot" style="color: #00d9ff;">📷 Download Snapshot</a></p>
    </div>
</body>
</html>"""
        response = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)


class StreamingServer:
    """MJPEG streaming server."""

    def __init__(
        self,
        camera: "Camera",
        port: int = 8080,
        stream_path: str = "/stream",
        jpeg_quality: int = 85,
        stream_fps: int = 15,
    ):
        self.camera = camera
        self.port = port
        self.stream_path = stream_path
        self.jpeg_quality = jpeg_quality
        self.stream_fps = stream_fps
        self._server = None
        self._thread = None

    def start(self):
        """Start the streaming server."""
        # Configure handler class
        StreamingHandler.camera = self.camera
        StreamingHandler.jpeg_quality = self.jpeg_quality
        StreamingHandler.stream_fps = self.stream_fps
        StreamingHandler.stream_path = self.stream_path

        self._server = HTTPServer(("0.0.0.0", self.port), StreamingHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

        logger.info(f"Streaming server started on http://0.0.0.0:{self.port}")
        logger.info(f"  - Stream: http://0.0.0.0:{self.port}{self.stream_path}")
        logger.info(f"  - Snapshot: http://0.0.0.0:{self.port}/snapshot")

    def stop(self):
        """Stop the streaming server."""
        if self._server:
            self._server.shutdown()
            self._server = None
        logger.info("Streaming server stopped")
