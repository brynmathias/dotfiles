"""
Metrics exporters for observability stacks.

Supports:
- Prometheus (pull-based metrics)
- InfluxDB (push-based time-series)
- StatsD (push to Graphite/Datadog)

All metrics from Garden Sentinel can be exported to your preferred
monitoring stack for dashboards, alerting, and long-term storage.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Type of metric."""
    COUNTER = "counter"      # Monotonically increasing
    GAUGE = "gauge"          # Can go up and down
    HISTOGRAM = "histogram"  # Distribution of values
    SUMMARY = "summary"      # Like histogram but calculates quantiles


@dataclass
class MetricValue:
    """A single metric value with labels."""
    name: str
    value: float
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: Optional[float] = None
    help_text: str = ""
    unit: str = ""


class MetricsRegistry:
    """
    Central registry for all metrics.

    Collectors register metrics here, exporters read from here.
    """

    def __init__(self):
        self._metrics: Dict[str, MetricValue] = {}
        self._counters: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, list] = defaultdict(list)

    def gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        help_text: str = "",
        unit: str = "",
    ):
        """Set a gauge metric (can go up or down)."""
        key = self._make_key(name, labels)
        self._metrics[key] = MetricValue(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            labels=labels or {},
            timestamp=time.time(),
            help_text=help_text,
            unit=unit,
        )

    def counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
        help_text: str = "",
    ):
        """Increment a counter metric (monotonically increasing)."""
        key = self._make_key(name, labels)
        self._counters[key] += value
        self._metrics[key] = MetricValue(
            name=name,
            value=self._counters[key],
            metric_type=MetricType.COUNTER,
            labels=labels or {},
            timestamp=time.time(),
            help_text=help_text,
        )

    def histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        help_text: str = "",
        buckets: Optional[list] = None,
    ):
        """Record a histogram value."""
        key = self._make_key(name, labels)
        self._histograms[key].append(value)

        # Keep last 1000 values for summary stats
        if len(self._histograms[key]) > 1000:
            self._histograms[key] = self._histograms[key][-1000:]

        # Calculate summary statistics
        values = self._histograms[key]
        self._metrics[key] = MetricValue(
            name=name,
            value=sum(values) / len(values),  # Mean for now
            metric_type=MetricType.HISTOGRAM,
            labels=labels or {},
            timestamp=time.time(),
            help_text=help_text,
        )

    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create a unique key for a metric with labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def get_all_metrics(self) -> list[MetricValue]:
        """Get all registered metrics."""
        return list(self._metrics.values())

    def clear(self):
        """Clear all metrics (useful for testing)."""
        self._metrics.clear()
        self._counters.clear()
        self._histograms.clear()


# Global registry
_registry = MetricsRegistry()


def get_registry() -> MetricsRegistry:
    """Get the global metrics registry."""
    return _registry


class MetricsExporter(ABC):
    """Abstract base class for metrics exporters."""

    @abstractmethod
    async def export(self, metrics: list[MetricValue]):
        """Export metrics to the backend."""
        pass

    @abstractmethod
    async def start(self):
        """Start the exporter."""
        pass

    @abstractmethod
    async def stop(self):
        """Stop the exporter."""
        pass


class PrometheusExporter(MetricsExporter):
    """
    Prometheus metrics exporter.

    Exposes metrics on an HTTP endpoint for Prometheus to scrape.
    Default port: 9090 (configurable)

    Metrics format:
    # HELP garden_sentinel_battery_voltage Battery voltage in volts
    # TYPE garden_sentinel_battery_voltage gauge
    garden_sentinel_battery_voltage{device="cam1"} 12.4
    """

    def __init__(
        self,
        port: int = 9100,
        host: str = "0.0.0.0",
        prefix: str = "garden_sentinel",
        registry: Optional[MetricsRegistry] = None,
    ):
        self.port = port
        self.host = host
        self.prefix = prefix
        self.registry = registry or get_registry()
        self._server = None
        self._app = None

    async def start(self):
        """Start the Prometheus HTTP server."""
        from aiohttp import web

        self._app = web.Application()
        self._app.router.add_get("/metrics", self._handle_metrics)
        self._app.router.add_get("/health", self._handle_health)

        runner = web.AppRunner(self._app)
        await runner.setup()
        self._server = web.TCPSite(runner, self.host, self.port)
        await self._server.start()

        logger.info(f"Prometheus exporter started on http://{self.host}:{self.port}/metrics")

    async def stop(self):
        """Stop the HTTP server."""
        if self._server:
            await self._server.stop()
        logger.info("Prometheus exporter stopped")

    async def _handle_metrics(self, request):
        """Handle /metrics endpoint."""
        from aiohttp import web

        metrics = self.registry.get_all_metrics()
        output = self._format_prometheus(metrics)

        return web.Response(
            text=output,
            content_type="text/plain; version=0.0.4; charset=utf-8",
        )

    async def _handle_health(self, request):
        """Handle /health endpoint."""
        from aiohttp import web
        return web.json_response({"status": "ok"})

    def _format_prometheus(self, metrics: list[MetricValue]) -> str:
        """Format metrics in Prometheus text format."""
        lines = []
        seen_names = set()

        for metric in metrics:
            full_name = f"{self.prefix}_{metric.name}"

            # Add HELP and TYPE only once per metric name
            if full_name not in seen_names:
                seen_names.add(full_name)
                if metric.help_text:
                    lines.append(f"# HELP {full_name} {metric.help_text}")
                lines.append(f"# TYPE {full_name} {metric.metric_type.value}")

            # Format labels
            if metric.labels:
                label_str = ",".join(
                    f'{k}="{v}"' for k, v in metric.labels.items()
                )
                lines.append(f"{full_name}{{{label_str}}} {metric.value}")
            else:
                lines.append(f"{full_name} {metric.value}")

        return "\n".join(lines) + "\n"

    async def export(self, metrics: list[MetricValue]):
        """Not used for Prometheus (pull-based)."""
        pass


class InfluxDBExporter(MetricsExporter):
    """
    InfluxDB metrics exporter.

    Pushes metrics to InfluxDB for time-series storage.
    Supports both InfluxDB 1.x and 2.x APIs.

    Line protocol format:
    garden_sentinel,device=cam1 battery_voltage=12.4 1234567890000000000
    """

    def __init__(
        self,
        url: str = "http://localhost:8086",
        token: Optional[str] = None,
        org: str = "garden-sentinel",
        bucket: str = "metrics",
        # For InfluxDB 1.x compatibility
        database: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        measurement: str = "garden_sentinel",
        registry: Optional[MetricsRegistry] = None,
        push_interval: float = 10.0,
    ):
        self.url = url.rstrip("/")
        self.token = token
        self.org = org
        self.bucket = bucket
        self.database = database
        self.username = username
        self.password = password
        self.measurement = measurement
        self.registry = registry or get_registry()
        self.push_interval = push_interval

        self._push_task: Optional[asyncio.Task] = None
        self._session = None

    async def start(self):
        """Start the push loop."""
        import aiohttp
        self._session = aiohttp.ClientSession()
        self._push_task = asyncio.create_task(self._push_loop())
        logger.info(f"InfluxDB exporter started, pushing to {self.url}")

    async def stop(self):
        """Stop the push loop."""
        if self._push_task:
            self._push_task.cancel()
            try:
                await self._push_task
            except asyncio.CancelledError:
                pass
        if self._session:
            await self._session.close()
        logger.info("InfluxDB exporter stopped")

    async def _push_loop(self):
        """Periodically push metrics to InfluxDB."""
        while True:
            try:
                await asyncio.sleep(self.push_interval)
                metrics = self.registry.get_all_metrics()
                if metrics:
                    await self.export(metrics)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error pushing to InfluxDB: {e}")

    async def export(self, metrics: list[MetricValue]):
        """Push metrics to InfluxDB."""
        if not self._session:
            return

        lines = self._format_line_protocol(metrics)
        if not lines:
            return

        body = "\n".join(lines)

        # Determine API version and endpoint
        if self.token:
            # InfluxDB 2.x API
            url = f"{self.url}/api/v2/write?org={self.org}&bucket={self.bucket}"
            headers = {
                "Authorization": f"Token {self.token}",
                "Content-Type": "text/plain",
            }
        else:
            # InfluxDB 1.x API
            db = self.database or self.bucket
            url = f"{self.url}/write?db={db}"
            headers = {"Content-Type": "text/plain"}
            if self.username:
                import base64
                auth = base64.b64encode(
                    f"{self.username}:{self.password}".encode()
                ).decode()
                headers["Authorization"] = f"Basic {auth}"

        try:
            async with self._session.post(url, data=body, headers=headers) as resp:
                if resp.status not in (200, 204):
                    text = await resp.text()
                    logger.error(f"InfluxDB write failed ({resp.status}): {text}")
                else:
                    logger.debug(f"Pushed {len(metrics)} metrics to InfluxDB")
        except Exception as e:
            logger.error(f"InfluxDB connection error: {e}")

    def _format_line_protocol(self, metrics: list[MetricValue]) -> list[str]:
        """Format metrics in InfluxDB line protocol."""
        lines = []

        for metric in metrics:
            # Tags from labels
            tags = ",".join(f"{k}={v}" for k, v in metric.labels.items())
            if tags:
                measurement = f"{self.measurement},{tags}"
            else:
                measurement = self.measurement

            # Field
            field = f"{metric.name}={metric.value}"

            # Timestamp in nanoseconds
            ts = int((metric.timestamp or time.time()) * 1_000_000_000)

            lines.append(f"{measurement} {field} {ts}")

        return lines


class StatsDExporter(MetricsExporter):
    """
    StatsD metrics exporter.

    Pushes metrics to StatsD (compatible with Graphite, Datadog, etc.)
    Uses UDP for low overhead.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8125,
        prefix: str = "garden_sentinel",
        registry: Optional[MetricsRegistry] = None,
        push_interval: float = 10.0,
    ):
        self.host = host
        self.port = port
        self.prefix = prefix
        self.registry = registry or get_registry()
        self.push_interval = push_interval

        self._push_task: Optional[asyncio.Task] = None
        self._socket = None

    async def start(self):
        """Start the push loop."""
        import socket
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._push_task = asyncio.create_task(self._push_loop())
        logger.info(f"StatsD exporter started, pushing to {self.host}:{self.port}")

    async def stop(self):
        """Stop the push loop."""
        if self._push_task:
            self._push_task.cancel()
            try:
                await self._push_task
            except asyncio.CancelledError:
                pass
        if self._socket:
            self._socket.close()
        logger.info("StatsD exporter stopped")

    async def _push_loop(self):
        """Periodically push metrics to StatsD."""
        while True:
            try:
                await asyncio.sleep(self.push_interval)
                metrics = self.registry.get_all_metrics()
                if metrics:
                    await self.export(metrics)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error pushing to StatsD: {e}")

    async def export(self, metrics: list[MetricValue]):
        """Push metrics to StatsD."""
        if not self._socket:
            return

        for metric in metrics:
            # Build metric name with labels as tags (DogStatsD format)
            name = f"{self.prefix}.{metric.name}"

            # StatsD type
            type_char = {
                MetricType.COUNTER: "c",
                MetricType.GAUGE: "g",
                MetricType.HISTOGRAM: "h",
                MetricType.SUMMARY: "ms",
            }.get(metric.metric_type, "g")

            # Format: metric.name:value|type|#tag1:val1,tag2:val2
            msg = f"{name}:{metric.value}|{type_char}"

            if metric.labels:
                tags = ",".join(f"{k}:{v}" for k, v in metric.labels.items())
                msg += f"|#{tags}"

            try:
                self._socket.sendto(msg.encode(), (self.host, self.port))
            except Exception as e:
                logger.debug(f"StatsD send error: {e}")


class GardenSentinelMetricsCollector:
    """
    Collects Garden Sentinel specific metrics and registers them.

    Call update_* methods to push new values, they'll be exported
    by whatever exporter is configured.
    """

    def __init__(
        self,
        device_id: str,
        registry: Optional[MetricsRegistry] = None,
    ):
        self.device_id = device_id
        self.registry = registry or get_registry()
        self._labels = {"device": device_id}

    # Battery metrics
    def update_battery_voltage(self, voltage: float):
        self.registry.gauge(
            "battery_voltage",
            voltage,
            self._labels,
            help_text="Battery voltage in volts",
            unit="volts",
        )

    def update_battery_current(self, current: float):
        self.registry.gauge(
            "battery_current",
            current,
            self._labels,
            help_text="Battery current in amps (negative=discharging)",
            unit="amps",
        )

    def update_battery_percentage(self, percentage: float):
        self.registry.gauge(
            "battery_percentage",
            percentage,
            self._labels,
            help_text="Battery charge percentage",
            unit="percent",
        )

    def update_battery_power(self, power: float):
        self.registry.gauge(
            "battery_power",
            power,
            self._labels,
            help_text="Battery power consumption in watts",
            unit="watts",
        )

    # System metrics
    def update_cpu_temperature(self, temp: float):
        self.registry.gauge(
            "cpu_temperature",
            temp,
            self._labels,
            help_text="CPU temperature in Celsius",
            unit="celsius",
        )

    def update_cpu_usage(self, usage: float):
        self.registry.gauge(
            "cpu_usage",
            usage,
            self._labels,
            help_text="CPU usage percentage",
            unit="percent",
        )

    def update_memory_usage(self, usage: float):
        self.registry.gauge(
            "memory_usage",
            usage,
            self._labels,
            help_text="Memory usage percentage",
            unit="percent",
        )

    def update_disk_usage(self, usage: float):
        self.registry.gauge(
            "disk_usage",
            usage,
            self._labels,
            help_text="Disk usage percentage",
            unit="percent",
        )

    def update_throttled(self, is_throttled: bool):
        self.registry.gauge(
            "cpu_throttled",
            1.0 if is_throttled else 0.0,
            self._labels,
            help_text="CPU throttling status (1=throttled)",
        )

    # Network metrics
    def update_wifi_signal(self, signal_percent: float):
        self.registry.gauge(
            "wifi_signal",
            signal_percent,
            self._labels,
            help_text="WiFi signal strength percentage",
            unit="percent",
        )

    def update_server_latency(self, latency_ms: float):
        self.registry.gauge(
            "server_latency",
            latency_ms,
            self._labels,
            help_text="Latency to server in milliseconds",
            unit="milliseconds",
        )

    # Camera metrics
    def update_camera_fps(self, fps: float):
        self.registry.gauge(
            "camera_fps",
            fps,
            self._labels,
            help_text="Camera frames per second",
        )

    def update_camera_errors(self, count: int):
        self.registry.gauge(
            "camera_errors",
            count,
            self._labels,
            help_text="Camera error count",
        )

    # Detection metrics
    def increment_detections(self, predator_type: str):
        labels = {**self._labels, "predator_type": predator_type}
        self.registry.counter(
            "detections_total",
            1,
            labels,
            help_text="Total predator detections",
        )

    def increment_alerts(self, threat_level: str):
        labels = {**self._labels, "threat_level": threat_level}
        self.registry.counter(
            "alerts_total",
            1,
            labels,
            help_text="Total alerts triggered",
        )

    def increment_sprays(self):
        self.registry.counter(
            "sprays_total",
            1,
            self._labels,
            help_text="Total spray activations",
        )

    def update_inference_time(self, time_ms: float):
        self.registry.histogram(
            "inference_time_ms",
            time_ms,
            self._labels,
            help_text="Model inference time in milliseconds",
        )

    # Health status
    def update_health_status(self, status: str):
        """Update overall health status (healthy=1, warning=0.5, critical=0)."""
        status_value = {
            "healthy": 1.0,
            "warning": 0.5,
            "critical": 0.0,
            "offline": -1.0,
        }.get(status.lower(), 0.0)

        self.registry.gauge(
            "health_status",
            status_value,
            self._labels,
            help_text="Device health status (1=healthy, 0.5=warning, 0=critical)",
        )


def create_exporter_from_config(config: dict) -> Optional[MetricsExporter]:
    """
    Create a metrics exporter from configuration.

    Config example:
    {
        "type": "prometheus",  # or "influxdb", "statsd"
        "port": 9100,

        # For InfluxDB:
        "url": "http://localhost:8086",
        "token": "xxx",
        "org": "garden-sentinel",
        "bucket": "metrics",

        # For StatsD:
        "host": "localhost",
        "port": 8125,
    }
    """
    exporter_type = config.get("type", "").lower()

    if exporter_type == "prometheus":
        return PrometheusExporter(
            port=config.get("port", 9100),
            host=config.get("host", "0.0.0.0"),
            prefix=config.get("prefix", "garden_sentinel"),
        )

    elif exporter_type == "influxdb":
        return InfluxDBExporter(
            url=config.get("url", "http://localhost:8086"),
            token=config.get("token"),
            org=config.get("org", "garden-sentinel"),
            bucket=config.get("bucket", "metrics"),
            database=config.get("database"),
            username=config.get("username"),
            password=config.get("password"),
            push_interval=config.get("push_interval", 10.0),
        )

    elif exporter_type == "statsd":
        return StatsDExporter(
            host=config.get("host", "localhost"),
            port=config.get("port", 8125),
            prefix=config.get("prefix", "garden_sentinel"),
            push_interval=config.get("push_interval", 10.0),
        )

    else:
        logger.warning(f"Unknown metrics exporter type: {exporter_type}")
        return None
