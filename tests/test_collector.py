"""Tests for the nvidia-smi CSV parser and the live collector.

These tests run with no GPU and no network access: the parser is fed an embedded
multi-GPU fixture, and the live collector is pointed at a non-existent binary to
exercise graceful degradation.
"""

from __future__ import annotations

from gpu_exporter.collector import (
    GPUReading,
    collect,
    nvidia_smi_csv,
    parse_csv,
)

# A two-GPU fixture resembling real `nvidia-smi --query-gpu=... --format=csv`
# output, including the bracketed units nvidia-smi appends to headers.
SAMPLE_CSV = """\
index, uuid, name, utilization.gpu [%], memory.used [MiB], memory.total [MiB], temperature.gpu, power.draw [W], power.limit [W], fan.speed [%], pcie.link.tx [KiB/s], pcie.link.rx [KiB/s]
0, GPU-abc123, NVIDIA GeForce RTX 4090, 50 %, 8000 MiB, 24564 MiB, 62, 120.50 W, 450.00 W, 35 %, 1000 KiB/s, 2000 KiB/s
1, GPU-def456, NVIDIA A100-SXM4-40GB, 25 %, 4096 MiB, 40960 MiB, 55, 85.25 W, 400.00 W, 0 %, 512 KiB/s, 768 KiB/s
"""


def test_parse_csv_returns_two_gpus() -> None:
    readings = parse_csv(SAMPLE_CSV)
    assert len(readings) == 2


def test_parse_csv_gpu0_values() -> None:
    readings = parse_csv(SAMPLE_CSV)
    gpu0 = readings[0]
    assert isinstance(gpu0, GPUReading)
    assert gpu0.index == 0
    assert gpu0.uuid == "GPU-abc123"
    assert gpu0.name == "NVIDIA GeForce RTX 4090"
    assert gpu0.utilization_percent == 50.0
    # 8000 MiB -> 8000 * 1024^2 bytes
    assert gpu0.memory_used_bytes == 8000 * 1024 ** 2
    assert gpu0.memory_total_bytes == 24564 * 1024 ** 2
    assert gpu0.temperature_celsius == 62.0
    assert gpu0.power_draw_watts == 120.50
    assert gpu0.power_limit_watts == 450.00
    assert gpu0.fan_speed_percent == 35.0
    # 1000 KiB/s -> 1000 * 1024 bytes
    assert gpu0.pcie_tx_bytes == 1000 * 1024
    assert gpu0.pcie_rx_bytes == 2000 * 1024


def test_parse_csv_gpu1_values() -> None:
    readings = parse_csv(SAMPLE_CSV)
    gpu1 = readings[1]
    assert gpu1.index == 1
    assert gpu1.uuid == "GPU-def456"
    assert gpu1.name == "NVIDIA A100-SXM4-40GB"
    assert gpu1.utilization_percent == 25.0
    assert gpu1.memory_used_bytes == 4096 * 1024 ** 2
    assert gpu1.memory_total_bytes == 40960 * 1024 ** 2
    assert gpu1.temperature_celsius == 55.0
    assert gpu1.power_draw_watts == 85.25
    assert gpu1.power_limit_watts == 400.00
    assert gpu1.fan_speed_percent == 0.0
    assert gpu1.pcie_tx_bytes == 512 * 1024
    assert gpu1.pcie_rx_bytes == 768 * 1024


def test_parse_csv_labels() -> None:
    readings = parse_csv(SAMPLE_CSV)
    assert readings[0].as_labels() == {
        "index": "0",
        "uuid": "GPU-abc123",
        "name": "NVIDIA GeForce RTX 4090",
    }


def test_parse_empty_csv() -> None:
    assert parse_csv("") == []
    assert parse_csv("index, uuid, name\n") == []


def test_collect_with_raw_text_reports_up() -> None:
    readings, up = collect(raw_text=SAMPLE_CSV)
    assert up is True
    assert len(readings) == 2


def test_collect_missing_binary_degrades() -> None:
    readings, up = collect(binary="this-binary-does-not-exist-xyz")
    assert up is False
    assert readings == []


def test_nvidia_smi_csv_missing_binary_raises() -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        nvidia_smi_csv(binary="this-binary-does-not-exist-xyz")
