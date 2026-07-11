"""gpu-exporter: a Prometheus exporter for NVIDIA GPU metrics.

The package parses ``nvidia-smi`` output into structured metrics and exposes them
on a ``/metrics`` endpoint. It degrades gracefully (``gpu_collector_up`` -> 0)
when no GPU or ``nvidia-smi`` is present, so it runs anywhere.
"""

from __future__ import annotations

from .app import DEFAULT_INTERVAL, DEFAULT_PORT, Exporter
from .collector import GPUReading, collect, parse_csv
from .metrics import MetricsRegistry

__version__ = "1.0.0"

__all__ = [
    "Exporter",
    "GPUReading",
    "MetricsRegistry",
    "collect",
    "parse_csv",
    "DEFAULT_PORT",
    "DEFAULT_INTERVAL",
]
