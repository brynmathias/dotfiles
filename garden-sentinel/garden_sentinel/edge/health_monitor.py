"""
Health monitoring for Garden Sentinel edge devices.

Monitors:
- Battery voltage and current (INA219/ADS1115)
- CPU temperature and throttling
- Memory usage
- Disk space
- Network connectivity
- Camera status
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Callable, Awaitable

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Overall health status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class BatteryStatus:
    """Battery monitoring data."""
    voltage: float  # Volts
    current: Optional[float] = None  # Amps (negative = discharging)
    power: Optional[float] = None  # Watts
    percentage: Optional[float] = None  # 0-100%
    is_charging: bool = False
    time_remaining: Optional[float] = None  # seconds
    status: HealthStatus = HealthStatus.UNKNOWN

    def to_dict(self) -> dict:
        return {
            "voltage": round(self.voltage, 2),
            "current": round(self.current, 3) if self.current else None,
            "power": round(self.power, 2) if self.power else None,
            "percentage": round(self.percentage, 1) if self.percentage else None,
            "is_charging": self.is_charging,
            "time_remaining_minutes": round(self.time_remaining / 60, 1) if self.time_remaining else None,
            "status": self.status.value,
        }


@dataclass
class SystemStatus:
    """System health data."""
    cpu_temp: float  # Celsius
    cpu_usage: float  # 0-100%
    memory_used: float  # 0-100%
    memory_available_mb: float
    disk_used: float  # 0-100%
    disk_available_gb: float
    is_throttled: bool = False
    throttle_flags: list[str] = field(default_factory=list)
    uptime_seconds: float = 0
    status: HealthStatus = HealthStatus.UNKNOWN

    def to_dict(self) -> dict:
        return {
            "cpu_temp": round(self.cpu_temp, 1),
            "cpu_usage": round(self.cpu_usage, 1),
            "memory_used_percent": round(self.memory_used, 1),
            "memory_available_mb": round(self.memory_available_mb, 1),
            "disk_used_percent": round(self.disk_used, 1),
            "disk_available_gb": round(self.disk_available_gb, 2),
            "is_throttled": self.is_throttled,
            "throttle_flags": self.throttle_flags,
            "uptime_hours": round(self.uptime_seconds / 3600, 1),
            "status": self.status.value,
        }


@dataclass
class NetworkStatus:
    """Network health data."""
    is_connected: bool = False
    server_reachable: bool = False
    latency_ms: Optional[float] = None
    wifi_signal_dbm: Optional[int] = None
    wifi_signal_percent: Optional[float] = None
    ip_address: Optional[str] = None
    status: HealthStatus = HealthStatus.UNKNOWN

    def to_dict(self) -> dict:
        return {
            "is_connected": self.is_connected,
            "server_reachable": self.server_reachable,
            "latency_ms": round(self.latency_ms, 1) if self.latency_ms else None,
            "wifi_signal_dbm": self.wifi_signal_dbm,
            "wifi_signal_percent": round(self.wifi_signal_percent, 1) if self.wifi_signal_percent else None,
            "ip_address": self.ip_address,
            "status": self.status.value,
        }


@dataclass
class CameraStatus:
    """Camera health data."""
    is_available: bool = False
    fps: float = 0.0
    resolution: Optional[tuple[int, int]] = None
    last_frame_time: Optional[float] = None
    error_count: int = 0
    status: HealthStatus = HealthStatus.UNKNOWN

    def to_dict(self) -> dict:
        return {
            "is_available": self.is_available,
            "fps": round(self.fps, 1),
            "resolution": f"{self.resolution[0]}x{self.resolution[1]}" if self.resolution else None,
            "seconds_since_frame": round(time.time() - self.last_frame_time, 1) if self.last_frame_time else None,
            "error_count": self.error_count,
            "status": self.status.value,
        }


@dataclass
class HealthReport:
    """Complete health report."""
    device_id: str
    timestamp: datetime
    battery: Optional[BatteryStatus] = None
    system: Optional[SystemStatus] = None
    network: Optional[NetworkStatus] = None
    camera: Optional[CameraStatus] = None
    overall_status: HealthStatus = HealthStatus.UNKNOWN
    alerts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "timestamp": self.timestamp.isoformat(),
            "battery": self.battery.to_dict() if self.battery else None,
            "system": self.system.to_dict() if self.system else None,
            "network": self.network.to_dict() if self.network else None,
            "camera": self.camera.to_dict() if self.camera else None,
            "overall_status": self.overall_status.value,
            "alerts": self.alerts,
        }


class BatteryMonitor(ABC):
    """Abstract base class for battery monitors."""

    @abstractmethod
    async def read(self) -> BatteryStatus:
        """Read current battery status."""
        pass


class INA219Monitor(BatteryMonitor):
    """
    Battery monitor using INA219 power monitor IC.

    Wiring:
    - VCC -> 3.3V
    - GND -> GND
    - SDA -> GPIO 2 (SDA)
    - SCL -> GPIO 3 (SCL)
    - VIN+ -> Battery+ (through shunt)
    - VIN- -> Load+

    The INA219 measures:
    - Shunt voltage (current through shunt resistor)
    - Bus voltage (load voltage)
    - Calculates current and power
    """

    def __init__(
        self,
        i2c_address: int = 0x40,
        shunt_ohms: float = 0.1,
        max_expected_amps: float = 3.2,
        battery_full_voltage: float = 12.6,  # 3S LiPo full
        battery_empty_voltage: float = 9.6,  # 3S LiPo empty (3.2V/cell)
        battery_capacity_ah: float = 5.0,  # Battery capacity in Ah
    ):
        self.i2c_address = i2c_address
        self.shunt_ohms = shunt_ohms
        self.max_expected_amps = max_expected_amps
        self.battery_full_voltage = battery_full_voltage
        self.battery_empty_voltage = battery_empty_voltage
        self.battery_capacity_ah = battery_capacity_ah

        self._ina = None
        self._init_sensor()

    def _init_sensor(self):
        """Initialize the INA219 sensor."""
        try:
            from ina219 import INA219, DeviceRangeError
            self._ina = INA219(
                self.shunt_ohms,
                self.max_expected_amps,
                address=self.i2c_address,
            )
            self._ina.configure()
            logger.info(f"INA219 initialized at address 0x{self.i2c_address:02x}")
        except ImportError:
            logger.warning("ina219 library not installed, battery monitoring disabled")
        except Exception as e:
            logger.error(f"Failed to initialize INA219: {e}")

    async def read(self) -> BatteryStatus:
        """Read battery status from INA219."""
        if not self._ina:
            return BatteryStatus(voltage=0, status=HealthStatus.UNKNOWN)

        try:
            voltage = self._ina.voltage()
            current = self._ina.current() / 1000  # mA to A
            power = self._ina.power() / 1000  # mW to W

            # Calculate percentage based on voltage
            voltage_range = self.battery_full_voltage - self.battery_empty_voltage
            percentage = max(0, min(100,
                (voltage - self.battery_empty_voltage) / voltage_range * 100
            ))

            # Determine if charging (positive current = charging)
            is_charging = current > 0.05

            # Estimate time remaining
            time_remaining = None
            if not is_charging and current < -0.1:
                # Discharging - estimate time based on current draw
                remaining_ah = (percentage / 100) * self.battery_capacity_ah
                time_remaining = (remaining_ah / abs(current)) * 3600  # seconds

            # Determine status
            if percentage < 10:
                status = HealthStatus.CRITICAL
            elif percentage < 25:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.HEALTHY

            return BatteryStatus(
                voltage=voltage,
                current=current,
                power=power,
                percentage=percentage,
                is_charging=is_charging,
                time_remaining=time_remaining,
                status=status,
            )

        except Exception as e:
            logger.error(f"Error reading INA219: {e}")
            return BatteryStatus(voltage=0, status=HealthStatus.UNKNOWN)


class ADS1115Monitor(BatteryMonitor):
    """
    Battery monitor using ADS1115 ADC with voltage divider.

    Wiring:
    - VDD -> 3.3V
    - GND -> GND
    - SDA -> GPIO 2 (SDA)
    - SCL -> GPIO 3 (SCL)
    - A0  -> Voltage divider output

    Voltage divider: Battery+ -> R1 -> ADC -> R2 -> GND
    """

    def __init__(
        self,
        i2c_address: int = 0x48,
        channel: int = 0,
        voltage_divider_ratio: float = 0.248,  # For 12V battery: 3.3/(10+3.3)
        battery_full_voltage: float = 12.6,
        battery_empty_voltage: float = 9.6,
    ):
        self.i2c_address = i2c_address
        self.channel = channel
        self.voltage_divider_ratio = voltage_divider_ratio
        self.battery_full_voltage = battery_full_voltage
        self.battery_empty_voltage = battery_empty_voltage

        self._ads = None
        self._init_sensor()

    def _init_sensor(self):
        """Initialize the ADS1115 sensor."""
        try:
            import board
            import busio
            import adafruit_ads1x15.ads1115 as ADS
            from adafruit_ads1x15.analog_in import AnalogIn

            i2c = busio.I2C(board.SCL, board.SDA)
            self._ads = ADS.ADS1115(i2c, address=self.i2c_address)
            self._channel = AnalogIn(self._ads, getattr(ADS, f'P{self.channel}'))
            logger.info(f"ADS1115 initialized at address 0x{self.i2c_address:02x}")
        except ImportError:
            logger.warning("adafruit-circuitpython-ads1x15 not installed")
        except Exception as e:
            logger.error(f"Failed to initialize ADS1115: {e}")

    async def read(self) -> BatteryStatus:
        """Read battery voltage from ADS1115."""
        if not self._ads:
            return BatteryStatus(voltage=0, status=HealthStatus.UNKNOWN)

        try:
            # Read ADC voltage and convert to actual battery voltage
            adc_voltage = self._channel.voltage
            battery_voltage = adc_voltage / self.voltage_divider_ratio

            # Calculate percentage
            voltage_range = self.battery_full_voltage - self.battery_empty_voltage
            percentage = max(0, min(100,
                (battery_voltage - self.battery_empty_voltage) / voltage_range * 100
            ))

            # Determine status
            if percentage < 10:
                status = HealthStatus.CRITICAL
            elif percentage < 25:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.HEALTHY

            return BatteryStatus(
                voltage=battery_voltage,
                percentage=percentage,
                status=status,
            )

        except Exception as e:
            logger.error(f"Error reading ADS1115: {e}")
            return BatteryStatus(voltage=0, status=HealthStatus.UNKNOWN)


class HealthMonitor:
    """
    Monitors edge device health and reports to server.

    Tracks:
    - Battery status (voltage, current, remaining time)
    - System status (CPU temp, memory, disk, throttling)
    - Network status (connectivity, latency, WiFi signal)
    - Camera status (FPS, errors)

    Optionally exports metrics to Prometheus/InfluxDB via metrics_collector.
    """

    def __init__(
        self,
        device_id: str,
        battery_monitor: Optional[BatteryMonitor] = None,
        server_url: Optional[str] = None,
        check_interval: float = 30.0,
        metrics_collector=None,  # GardenSentinelMetricsCollector
        # Thresholds
        cpu_temp_warning: float = 70.0,
        cpu_temp_critical: float = 80.0,
        memory_warning: float = 80.0,
        memory_critical: float = 95.0,
        disk_warning: float = 80.0,
        disk_critical: float = 95.0,
    ):
        self.device_id = device_id
        self.battery_monitor = battery_monitor
        self.server_url = server_url
        self.check_interval = check_interval
        self.metrics_collector = metrics_collector

        # Thresholds
        self.cpu_temp_warning = cpu_temp_warning
        self.cpu_temp_critical = cpu_temp_critical
        self.memory_warning = memory_warning
        self.memory_critical = memory_critical
        self.disk_warning = disk_warning
        self.disk_critical = disk_critical

        # State
        self._last_report: Optional[HealthReport] = None
        self._camera_fps_samples: list[float] = []
        self._camera_error_count = 0
        self._camera_last_frame_time: Optional[float] = None
        self._camera_resolution: Optional[tuple[int, int]] = None

        # Callbacks
        self._on_alert: Optional[Callable[[str, HealthStatus], Awaitable[None]]] = None
        self._on_report: Optional[Callable[[HealthReport], Awaitable[None]]] = None

        # Background task
        self._monitor_task: Optional[asyncio.Task] = None

    def set_callbacks(
        self,
        on_alert: Optional[Callable[[str, HealthStatus], Awaitable[None]]] = None,
        on_report: Optional[Callable[[HealthReport], Awaitable[None]]] = None,
    ):
        """Set event callbacks."""
        self._on_alert = on_alert
        self._on_report = on_report

    async def start(self):
        """Start health monitoring."""
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Health monitor started")

    async def stop(self):
        """Stop health monitoring."""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitor stopped")

    def update_camera_stats(
        self,
        fps: Optional[float] = None,
        resolution: Optional[tuple[int, int]] = None,
        error: bool = False,
    ):
        """Update camera statistics from camera module."""
        self._camera_last_frame_time = time.time()

        if fps is not None:
            self._camera_fps_samples.append(fps)
            # Keep last 10 samples
            self._camera_fps_samples = self._camera_fps_samples[-10:]

        if resolution:
            self._camera_resolution = resolution

        if error:
            self._camera_error_count += 1

    async def _monitor_loop(self):
        """Main monitoring loop."""
        while True:
            try:
                report = await self.collect_health_report()
                self._last_report = report

                # Check for alerts
                await self._check_alerts(report)

                # Send report callback
                if self._on_report:
                    await self._on_report(report)

                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(self.check_interval)

    async def collect_health_report(self) -> HealthReport:
        """Collect complete health report."""
        alerts = []

        # Collect all health data
        battery = await self._collect_battery() if self.battery_monitor else None
        system = await self._collect_system()
        network = await self._collect_network()
        camera = self._collect_camera()

        # Determine overall status
        statuses = [
            s.status for s in [battery, system, network, camera]
            if s is not None
        ]

        if HealthStatus.CRITICAL in statuses:
            overall = HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            overall = HealthStatus.WARNING
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            overall = HealthStatus.HEALTHY
        else:
            overall = HealthStatus.UNKNOWN

        # Collect alert messages
        if battery and battery.status == HealthStatus.CRITICAL:
            alerts.append(f"Battery critical: {battery.percentage:.0f}%")
        elif battery and battery.status == HealthStatus.WARNING:
            alerts.append(f"Battery low: {battery.percentage:.0f}%")

        if system and system.status == HealthStatus.CRITICAL:
            if system.cpu_temp > self.cpu_temp_critical:
                alerts.append(f"CPU overheating: {system.cpu_temp:.1f}°C")
            if system.is_throttled:
                alerts.append(f"CPU throttled: {', '.join(system.throttle_flags)}")

        if network and not network.server_reachable:
            alerts.append("Server unreachable")

        if camera and camera.status != HealthStatus.HEALTHY:
            alerts.append(f"Camera issues: {camera.error_count} errors")

        report = HealthReport(
            device_id=self.device_id,
            timestamp=datetime.now(),
            battery=battery,
            system=system,
            network=network,
            camera=camera,
            overall_status=overall,
            alerts=alerts,
        )

        # Export metrics to Prometheus/InfluxDB if configured
        self._export_metrics(report)

        return report

    def _export_metrics(self, report: HealthReport):
        """Export health data to metrics collector (Prometheus/InfluxDB)."""
        if not self.metrics_collector:
            return

        mc = self.metrics_collector

        # Battery metrics
        if report.battery:
            mc.update_battery_voltage(report.battery.voltage)
            if report.battery.current is not None:
                mc.update_battery_current(report.battery.current)
            if report.battery.percentage is not None:
                mc.update_battery_percentage(report.battery.percentage)
            if report.battery.power is not None:
                mc.update_battery_power(report.battery.power)

        # System metrics
        if report.system:
            mc.update_cpu_temperature(report.system.cpu_temp)
            mc.update_cpu_usage(report.system.cpu_usage)
            mc.update_memory_usage(report.system.memory_used)
            mc.update_disk_usage(report.system.disk_used)
            mc.update_throttled(report.system.is_throttled)

        # Network metrics
        if report.network:
            if report.network.wifi_signal_percent is not None:
                mc.update_wifi_signal(report.network.wifi_signal_percent)
            if report.network.latency_ms is not None:
                mc.update_server_latency(report.network.latency_ms)

        # Camera metrics
        if report.camera:
            mc.update_camera_fps(report.camera.fps)
            mc.update_camera_errors(report.camera.error_count)

        # Overall health status
        mc.update_health_status(report.overall_status.value)

    async def _collect_battery(self) -> Optional[BatteryStatus]:
        """Collect battery status."""
        if not self.battery_monitor:
            return None
        return await self.battery_monitor.read()

    async def _collect_system(self) -> SystemStatus:
        """Collect system health data."""
        import psutil

        # CPU temperature
        cpu_temp = 0.0
        try:
            temp_path = Path("/sys/class/thermal/thermal_zone0/temp")
            if temp_path.exists():
                cpu_temp = int(temp_path.read_text().strip()) / 1000
        except Exception:
            pass

        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=0.1)

        # Memory
        mem = psutil.virtual_memory()
        memory_used = mem.percent
        memory_available_mb = mem.available / (1024 * 1024)

        # Disk
        disk = psutil.disk_usage("/")
        disk_used = disk.percent
        disk_available_gb = disk.free / (1024 * 1024 * 1024)

        # Throttling (Raspberry Pi specific)
        is_throttled = False
        throttle_flags = []
        try:
            import subprocess
            result = subprocess.run(
                ["vcgencmd", "get_throttled"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                throttled_hex = result.stdout.strip().split("=")[1]
                throttled = int(throttled_hex, 16)

                flag_meanings = {
                    0: "Under-voltage detected",
                    1: "Arm frequency capped",
                    2: "Currently throttled",
                    3: "Soft temperature limit active",
                    16: "Under-voltage has occurred",
                    17: "Arm frequency capping has occurred",
                    18: "Throttling has occurred",
                    19: "Soft temperature limit has occurred",
                }

                for bit, meaning in flag_meanings.items():
                    if throttled & (1 << bit):
                        throttle_flags.append(meaning)
                        if bit < 16:  # Current flags
                            is_throttled = True
        except Exception:
            pass

        # Uptime
        uptime = time.time() - psutil.boot_time()

        # Determine status
        if cpu_temp > self.cpu_temp_critical or memory_used > self.memory_critical or disk_used > self.disk_critical:
            status = HealthStatus.CRITICAL
        elif cpu_temp > self.cpu_temp_warning or memory_used > self.memory_warning or disk_used > self.disk_warning or is_throttled:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY

        return SystemStatus(
            cpu_temp=cpu_temp,
            cpu_usage=cpu_usage,
            memory_used=memory_used,
            memory_available_mb=memory_available_mb,
            disk_used=disk_used,
            disk_available_gb=disk_available_gb,
            is_throttled=is_throttled,
            throttle_flags=throttle_flags,
            uptime_seconds=uptime,
            status=status,
        )

    async def _collect_network(self) -> NetworkStatus:
        """Collect network health data."""
        import socket

        is_connected = False
        server_reachable = False
        latency_ms = None
        wifi_signal_dbm = None
        wifi_signal_percent = None
        ip_address = None

        # Check basic connectivity
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            is_connected = True
        except Exception:
            pass

        # Get IP address
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip_address = s.getsockname()[0]
            s.close()
        except Exception:
            pass

        # Check server connectivity
        if self.server_url and is_connected:
            try:
                import aiohttp
                start = time.time()
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                    async with session.get(f"{self.server_url}/health") as resp:
                        if resp.status == 200:
                            server_reachable = True
                            latency_ms = (time.time() - start) * 1000
            except Exception:
                pass

        # WiFi signal strength (Linux/Raspberry Pi)
        try:
            with open("/proc/net/wireless") as f:
                lines = f.readlines()
                if len(lines) >= 3:
                    # Parse signal level from third line
                    parts = lines[2].split()
                    if len(parts) >= 4:
                        signal = float(parts[3].rstrip('.'))
                        wifi_signal_dbm = int(signal)
                        # Convert to percentage (rough approximation)
                        # -30 dBm = excellent, -90 dBm = unusable
                        wifi_signal_percent = max(0, min(100, (wifi_signal_dbm + 90) * (100 / 60)))
        except Exception:
            pass

        # Determine status
        if not is_connected:
            status = HealthStatus.CRITICAL
        elif not server_reachable:
            status = HealthStatus.WARNING
        elif wifi_signal_percent is not None and wifi_signal_percent < 30:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY

        return NetworkStatus(
            is_connected=is_connected,
            server_reachable=server_reachable,
            latency_ms=latency_ms,
            wifi_signal_dbm=wifi_signal_dbm,
            wifi_signal_percent=wifi_signal_percent,
            ip_address=ip_address,
            status=status,
        )

    def _collect_camera(self) -> CameraStatus:
        """Collect camera health data."""
        avg_fps = 0.0
        if self._camera_fps_samples:
            avg_fps = sum(self._camera_fps_samples) / len(self._camera_fps_samples)

        is_available = self._camera_last_frame_time is not None
        if is_available:
            time_since_frame = time.time() - self._camera_last_frame_time
            if time_since_frame > 10:
                is_available = False

        # Determine status
        if not is_available or self._camera_error_count > 10:
            status = HealthStatus.CRITICAL
        elif self._camera_error_count > 3 or avg_fps < 5:
            status = HealthStatus.WARNING
        elif is_available:
            status = HealthStatus.HEALTHY
        else:
            status = HealthStatus.UNKNOWN

        return CameraStatus(
            is_available=is_available,
            fps=avg_fps,
            resolution=self._camera_resolution,
            last_frame_time=self._camera_last_frame_time,
            error_count=self._camera_error_count,
            status=status,
        )

    async def _check_alerts(self, report: HealthReport):
        """Check for alert conditions and trigger callbacks."""
        if not self._on_alert:
            return

        for alert_msg in report.alerts:
            await self._on_alert(alert_msg, report.overall_status)

    def get_last_report(self) -> Optional[HealthReport]:
        """Get the last collected health report."""
        return self._last_report


def create_battery_monitor(config: dict) -> Optional[BatteryMonitor]:
    """
    Create battery monitor from configuration.

    Config example:
    {
        "type": "ina219",  # or "ads1115"
        "i2c_address": 0x40,
        "battery_full_voltage": 12.6,
        "battery_empty_voltage": 9.6,
        "battery_capacity_ah": 5.0,
        # INA219 specific
        "shunt_ohms": 0.1,
        # ADS1115 specific
        "channel": 0,
        "voltage_divider_ratio": 0.248,
    }
    """
    monitor_type = config.get("type", "").lower()

    if monitor_type == "ina219":
        return INA219Monitor(
            i2c_address=config.get("i2c_address", 0x40),
            shunt_ohms=config.get("shunt_ohms", 0.1),
            max_expected_amps=config.get("max_expected_amps", 3.2),
            battery_full_voltage=config.get("battery_full_voltage", 12.6),
            battery_empty_voltage=config.get("battery_empty_voltage", 9.6),
            battery_capacity_ah=config.get("battery_capacity_ah", 5.0),
        )
    elif monitor_type == "ads1115":
        return ADS1115Monitor(
            i2c_address=config.get("i2c_address", 0x48),
            channel=config.get("channel", 0),
            voltage_divider_ratio=config.get("voltage_divider_ratio", 0.248),
            battery_full_voltage=config.get("battery_full_voltage", 12.6),
            battery_empty_voltage=config.get("battery_empty_voltage", 9.6),
        )
    else:
        logger.warning(f"Unknown battery monitor type: {monitor_type}")
        return None
