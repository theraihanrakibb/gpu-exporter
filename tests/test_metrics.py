"""Tests for the Prometheus metrics registry."""

from __future__ import annotations

from prometheus_client import CollectorRegistry

from gpu_exporter.collector import parse_csv
from gpu_exporter.metrics import MetricsRegistry

SAMPLE_CSV = """\
index, uuid, name, utilization.gpu [%], memory.used [MiB], memory.total [MiB], temperature.gpu, power.draw [W], power.limit [W], fan.speed [%], pcie.link.tx [KiB/s], pcie.link.rx [KiB/s]
0, GPU-abc123, NVIDIA GeForce RTX 4090, 50 %, 8000 MiB, 24564 MiB, 62, 120.50 W, 450.00 W, 35 %, 1000 KiB/s, 2000 KiB/s
1, GPU-def456, NVIDIA A100-SXM4-40GB, 25 %, 4096 MiB, 40960 MiB, 55, 85.25 W, 400.00 W, 0 %, 512 KiB/s, 768 KiB/s
"""


def _registry() -> CollectorRegistry:
    return CollectorRegistry()


def test_update_sets_per_gpu_metrics() -> None:
    registry = _registry()
    metrics = MetricsRegistry(registry)
    readings = parse_csv(SAMPLE_CSV)
    metrics.update(readings, up=True)

    labels0 = {"index": "0", "uuid": "GPU-abc123", "name": "NVIDIA GeForce RTX 4090"}
    labels1 = {"index": "1", "uuid": "GPU-def456", "name": "NVIDIA A100-SXM4-40GB"}

    assert registry.get_sample_value("gpu_utilization_percent", labels0) == 50.0
    assert registry.get_sample_value("gpu_utilization_percent", labels1) == 25.0
    assert registry.get_sample_value("gpu_memory_used_bytes", labels0) == 8000 * 1024 ** 2
    assert registry.get_sample_value("gpu_memory_total_bytes", labels1) == 40960 * 1024 ** 2
    assert registry.get_sample_value("gpu_temperature_celsius", labels0) == 62.0
    assert registry.get_sample_value("gpu_power_draw_watts", labels0) == 120.50
    assert registry.get_sample_value("gpu_power_limit_watts", labels1) == 400.00
    assert registry.get_sample_value("gpu_fan_speed_percent", labels0) == 35.0
    assert registry.get_sample_value("gpu_pcie_tx_bytes", labels0) == 1000 * 1024
    assert registry.get_sample_value("gpu_pcie_rx_bytes", labels1) == 768 * 1024


def test_update_sets_global_signals() -> None:
    registry = _registry()
    metrics = MetricsRegistry(registry)
    metrics.update(parse_csv(SAMPLE_CSV), up=True)
    assert registry.get_sample_value("gpu_count") == 2.0
    assert registry.get_sample_value("gpu_collector_up") == 1.0


def test_update_degraded_sets_collector_down() -> None:
    registry = _registry()
    metrics = MetricsRegistry(registry)
    metrics.update([], up=False)
    assert registry.get_sample_value("gpu_collector_up") == 0.0
    assert registry.get_sample_value("gpu_count") == 0.0


def test_update_clears_stale_series() -> None:
    registry = _registry()
    metrics = MetricsRegistry(registry)
    metrics.update(parse_csv(SAMPLE_CSV), up=True)
    # Re-collect with only one GPU; the other series must be removed.
    metrics.update(parse_csv(SAMPLE_CSV)[:1], up=True)
    labels1 = {"index": "1", "uuid": "GPU-def456", "name": "NVIDIA A100-SXM4-40GB"}
    assert registry.get_sample_value("gpu_utilization_percent", labels1) is None
    assert registry.get_sample_value("gpu_count") == 1.0
