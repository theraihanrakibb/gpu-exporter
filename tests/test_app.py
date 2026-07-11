"""Tests for the HTTP exposer (the /metrics endpoint)."""

from __future__ import annotations

import socket
import threading
import time
import urllib.request
from prometheus_client import CollectorRegistry

from gpu_exporter.app import Exporter

SAMPLE_CSV = """\
index, uuid, name, utilization.gpu [%], memory.used [MiB], memory.total [MiB], temperature.gpu, power.draw [W], power.limit [W], fan.speed [%], pcie.link.tx [KiB/s], pcie.link.rx [KiB/s]
0, GPU-abc123, NVIDIA GeForce RTX 4090, 50 %, 8000 MiB, 24564 MiB, 62, 120.50 W, 450.00 W, 35 %, 1000 KiB/s, 2000 KiB/s
"""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def test_metrics_endpoint_serves_readings() -> None:
    port = _free_port()
    registry = CollectorRegistry()
    exporter = Exporter(
        port=port, interval=1.0, collector_text=SAMPLE_CSV, registry=registry
    )
    exporter.start()
    try:
        # Give the HTTP server a moment to bind.
        time.sleep(0.3)
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as resp:
            body = resp.read().decode("utf-8")
    finally:
        exporter.stop()

    assert resp.status == 200
    assert "gpu_utilization_percent" in body
    assert "gpu_memory_used_bytes" in body
    assert "gpu_collector_up 1.0" in body
    assert "gpu_count 1.0" in body
    assert "GPU-abc123" in body


def test_metrics_endpoint_degraded_keeps_serving() -> None:
    port = _free_port()
    registry = CollectorRegistry()
    # Point at a missing binary so the collector degrades to up=0.
    exporter = Exporter(
        port=port,
        interval=1.0,
        binary="no-such-binary-xyz",
        registry=registry,
    )
    exporter.start()
    try:
        time.sleep(0.3)
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as resp:
            body = resp.read().decode("utf-8")
    finally:
        exporter.stop()

    assert resp.status == 200
    # Even degraded, the endpoint keeps serving and reports collector down.
    assert "gpu_collector_up 0.0" in body
