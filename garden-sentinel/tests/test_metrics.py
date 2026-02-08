"""
Tests for the shared metrics module.
"""

import pytest
import time
from garden_sentinel.shared.metrics import (
    MetricsRegistry,
    GardenSentinelMetricsCollector,
)


class TestMetricsRegistry:
    """Tests for MetricsRegistry class."""

    def test_registry_creation(self):
        registry = MetricsRegistry()
        assert registry is not None

    def test_counter(self):
        registry = MetricsRegistry()
        registry.counter("test_counter", 1)
        registry.counter("test_counter", 2)

        metrics = registry.get_all_metrics()
        assert "test_counter" in metrics
        assert metrics["test_counter"]["value"] == 3

    def test_counter_with_labels(self):
        registry = MetricsRegistry()
        registry.counter("requests", 1, {"method": "GET", "status": "200"})
        registry.counter("requests", 1, {"method": "POST", "status": "201"})
        registry.counter("requests", 2, {"method": "GET", "status": "200"})

        metrics = registry.get_all_metrics()
        # Should have separate entries for each label combination
        assert "requests" in metrics

    def test_gauge(self):
        registry = MetricsRegistry()
        registry.gauge("temperature", 25.5)
        registry.gauge("temperature", 26.0)

        metrics = registry.get_all_metrics()
        assert "temperature" in metrics
        assert metrics["temperature"]["value"] == 26.0  # Last value

    def test_histogram(self):
        registry = MetricsRegistry()
        registry.histogram("request_duration", 0.1)
        registry.histogram("request_duration", 0.2)
        registry.histogram("request_duration", 0.3)

        metrics = registry.get_all_metrics()
        assert "request_duration" in metrics

    def test_help_text(self):
        registry = MetricsRegistry()
        registry.counter("test_metric", 1, help_text="A test metric")

        metrics = registry.get_all_metrics()
        assert metrics["test_metric"]["help"] == "A test metric"


class TestGardenSentinelMetricsCollector:
    """Tests for GardenSentinelMetricsCollector class."""

    @pytest.fixture
    def collector(self):
        registry = MetricsRegistry()
        return GardenSentinelMetricsCollector(
            registry=registry,
            device_id="test-device",
        )

    def test_collector_creation(self, collector):
        assert collector is not None
        assert collector._labels["device_id"] == "test-device"

    def test_increment_frames_processed(self, collector):
        collector.increment_frames_processed()
        collector.increment_frames_processed()

        metrics = collector.registry.get_all_metrics()
        assert "frames_processed_total" in metrics
        assert metrics["frames_processed_total"]["value"] == 2

    def test_increment_detections(self, collector):
        collector.increment_detections("fox")
        collector.increment_detections("fox")
        collector.increment_detections("cat")

        metrics = collector.registry.get_all_metrics()
        assert "detections_total" in metrics

    def test_increment_sprays(self, collector):
        collector.increment_sprays("fox")

        metrics = collector.registry.get_all_metrics()
        assert "sprays_total" in metrics

    def test_increment_deterred(self, collector):
        collector.increment_deterred("fox")

        metrics = collector.registry.get_all_metrics()
        assert "deterred_total" in metrics

    def test_set_battery_voltage(self, collector):
        collector.set_battery_voltage(12.6)

        metrics = collector.registry.get_all_metrics()
        assert "battery_voltage" in metrics
        assert metrics["battery_voltage"]["value"] == 12.6

    def test_set_battery_percent(self, collector):
        collector.set_battery_percent(85.0)

        metrics = collector.registry.get_all_metrics()
        assert "battery_percent" in metrics

    def test_set_cpu_temperature(self, collector):
        collector.set_cpu_temperature(55.0)

        metrics = collector.registry.get_all_metrics()
        assert "cpu_temperature_celsius" in metrics

    def test_set_memory_usage(self, collector):
        collector.set_memory_usage(2048, 4096)

        metrics = collector.registry.get_all_metrics()
        assert "memory_used_bytes" in metrics
        assert "memory_total_bytes" in metrics

    def test_record_inference_time(self, collector):
        collector.record_inference_time(0.05)
        collector.record_inference_time(0.06)

        metrics = collector.registry.get_all_metrics()
        assert "inference_duration_seconds" in metrics

    def test_record_network_latency(self, collector):
        collector.record_network_latency(0.025)

        metrics = collector.registry.get_all_metrics()
        assert "network_latency_seconds" in metrics


class TestMetricsLabels:
    """Tests for metrics label handling."""

    def test_labels_in_counter(self):
        registry = MetricsRegistry()
        collector = GardenSentinelMetricsCollector(
            registry=registry,
            device_id="cam-01",
            location="front_garden",
        )

        collector.increment_detections("fox")

        # Labels should include device_id, location, and predator_type
        metrics = registry.get_all_metrics()
        # Check that the metric was recorded with labels
        assert "detections_total" in metrics

    def test_different_devices(self):
        registry = MetricsRegistry()

        collector1 = GardenSentinelMetricsCollector(
            registry=registry,
            device_id="cam-01",
        )
        collector2 = GardenSentinelMetricsCollector(
            registry=registry,
            device_id="cam-02",
        )

        collector1.increment_frames_processed()
        collector1.increment_frames_processed()
        collector2.increment_frames_processed()

        # Both should be tracked in the same registry
        metrics = registry.get_all_metrics()
        assert "frames_processed_total" in metrics
