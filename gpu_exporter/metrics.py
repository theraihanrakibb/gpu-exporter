"""Prometheus metric registration and updating for GPU readings.

All per-GPU gauges carry the ``index`` / ``uuid`` / ``name`` label triple so that
a single exporter can distinguish multiple GPUs in a fleet. The ``collector_up``
and ``gpu_count`` gauges are global signals useful for alerting and dashboards.
"""

from __future__ import annotations

from typing import Optional

from prometheus_client import CollectorRegistry, Gauge

from .collector import GPUReading

_GPU_LABELS = ("index", "uuid", "name")


class MetricsRegistry:
    """Owns the Prometheus gauges and applies fresh GPU readings to them."""

    def __init__(self, registry: Optional[CollectorRegistry] = None) -> None:
        """Create the gauges on ``registry`` (defaults to the global registry)."""
        self.registry = registry

        self.gpu_util = Gauge(
            "gpu_utilization_percent",
            "GPU core utilization as a percentage (0-100).",
            _GPU_LABELS,
            registry=self.registry,
        )
        self.memory_used = Gauge(
            "gpu_memory_used_bytes",
            "Memory currently used by the GPU in bytes.",
            _GPU_LABELS,
            registry=self.registry,
        )
        self.memory_total = Gauge(
            "gpu_memory_total_bytes",
            "Total installed GPU memory in bytes.",
            _GPU_LABELS,
            registry=self.registry,
        )
        self.temperature = Gauge(
            "gpu_temperature_celsius",
            "GPU temperature in degrees Celsius.",
            _GPU_LABELS,
            registry=self.registry,
        )
        self.power_draw = Gauge(
            "gpu_power_draw_watts",
            "Instantaneous power draw of the GPU in watts.",
            _GPU_LABELS,
            registry=self.registry,
        )
        self.power_limit = Gauge(
            "gpu_power_limit_watts",
            "Configured power limit of the GPU in watts.",
            _GPU_LABELS,
            registry=self.registry,
        )
        self.fan_speed = Gauge(
            "gpu_fan_speed_percent",
            "GPU fan speed as a percentage (0-100).",
            _GPU_LABELS,
            registry=self.registry,
        )
        self.pcie_tx = Gauge(
            "gpu_pcie_tx_bytes",
            "PCIe transmit throughput counter in bytes (optional).",
            _GPU_LABELS,
            registry=self.registry,
        )
        self.pcie_rx = Gauge(
            "gpu_pcie_rx_bytes",
            "PCIe receive throughput counter in bytes (optional).",
            _GPU_LABELS,
            registry=self.registry,
        )
        self.gpu_count = Gauge(
            "gpu_count",
            "Number of GPUs discovered in the last successful collection.",
            registry=self.registry,
        )
        self.collector_up = Gauge(
            "gpu_collector_up",
            "1 if the last collection succeeded, 0 if nvidia-smi was missing or errored.",
            registry=self.registry,
        )

    def update(self, readings: list[GPUReading], up: bool) -> None:
        """Apply a fresh collection result to all gauges.

        Labeled gauges are cleared first so that GPUs that disappear between
        collections do not leave stale series behind.
        """
        self.collector_up.set(1.0 if up else 0.0)
        self.gpu_count.set(float(len(readings)))

        for gauge in (
            self.gpu_util,
            self.memory_used,
            self.memory_total,
            self.temperature,
            self.power_draw,
            self.power_limit,
            self.fan_speed,
            self.pcie_tx,
            self.pcie_rx,
        ):
            gauge.clear()

        for reading in readings:
            labels = reading.as_labels()
            self.gpu_util.labels(**labels).set(reading.utilization_percent)
            self.memory_used.labels(**labels).set(reading.memory_used_bytes)
            self.memory_total.labels(**labels).set(reading.memory_total_bytes)
            self.temperature.labels(**labels).set(reading.temperature_celsius)
            self.power_draw.labels(**labels).set(reading.power_draw_watts)
            self.power_limit.labels(**labels).set(reading.power_limit_watts)
            self.fan_speed.labels(**labels).set(reading.fan_speed_percent)
            if reading.pcie_tx_bytes is not None:
                self.pcie_tx.labels(**labels).set(reading.pcie_tx_bytes)
            if reading.pcie_rx_bytes is not None:
                self.pcie_rx.labels(**labels).set(reading.pcie_rx_bytes)
