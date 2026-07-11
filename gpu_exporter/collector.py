"""NVIDIA GPU metric collection and parsing.

This module parses the CSV output of ``nvidia-smi --query-gpu=... --format=csv``
into structured :class:`GPUReading` dataclasses. It is intentionally free of any
GPU-specific native dependencies (no ``pynvml`` / ``torch``); it only shells out
to the ``nvidia-smi`` binary when actually collecting from a live system.

The pure parsing function :func:`parse_csv` accepts a raw CSV string so it can be
exercised in unit tests without a GPU or the ``nvidia-smi`` binary.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Fields requested from nvidia-smi. The exact set drives both the live query and
# the column detection in the parser, so they must stay in sync.
QUERY_FIELDS = (
    "index,uuid,name,utilization.gpu,memory.used,memory.total,"
    "temperature.gpu,power.draw,power.limit,fan.speed,"
    "pcie.link.tx,pcie.link.rx"
)

# Canonical field name -> substring(s) used to locate the column in the header.
_FIELD_KEYWORDS: dict[str, tuple[str, ...]] = {
    "index": ("index",),
    "uuid": ("uuid",),
    "name": ("name",),
    "utilization": ("utilization.gpu",),
    "mem_used": ("memory.used",),
    "mem_total": ("memory.total",),
    "temperature": ("temperature.gpu",),
    "power_draw": ("power.draw",),
    "power_limit": ("power.limit",),
    "fan_speed": ("fan.speed",),
    "pcie_tx": ("pcie.link.tx",),
    "pcie_rx": ("pcie.link.rx",),
}

# Fields whose raw values carry a byte-oriented unit that must be normalised to
# bytes before being exposed as a Prometheus metric.
_BYTE_FIELDS = {"mem_used", "mem_total", "pcie_tx", "pcie_rx"}


@dataclass
class GPUReading:
    """A single structured reading for one physical GPU.

    Memory and PCIe counters are stored in bytes; percentage / temperature /
    power values keep their natural units. ``None`` for an optional counter means
    the value was reported as unsupported by the driver.
    """

    index: int
    uuid: str
    name: str
    utilization_percent: float
    memory_used_bytes: int
    memory_total_bytes: int
    temperature_celsius: float
    power_draw_watts: float
    power_limit_watts: float
    fan_speed_percent: float
    pcie_tx_bytes: Optional[int] = None
    pcie_rx_bytes: Optional[int] = None

    def as_labels(self) -> dict[str, str]:
        """Return the Prometheus label set that identifies this GPU."""
        return {"index": str(self.index), "uuid": self.uuid, "name": self.name}


_UNSUPPORTED = {"", "[not supported]", "n/a", "none", "null"}


def _clean_numeric(token: str) -> Optional[float]:
    """Strip unit suffixes and return a float, or ``None`` if unsupported."""
    token = (token or "").strip()
    if token.lower() in _UNSUPPORTED:
        return None
    # Keep only digits, sign, decimal point and exponent markers.
    cleaned = re.sub(r"[^0-9eE.\-]", "", token)
    if cleaned in ("", "-", ".", "-."):
        return None
    try:
        return float(cleaned)
    except ValueError:
        logger.debug("Could not parse numeric token %r", token)
        return None


def _unit_multiplier(header_name: str) -> float:
    """Return the bytes-per-unit factor implied by a header's bracketed unit.

    nvidia-smi reports memory in MiB/GiB and PCIe bandwidth in KiB/s or MiB/s.
    We normalise everything to bytes.
    """
    name = header_name.lower()
    if "gib" in name:
        return 1024 ** 3
    if "mib" in name:
        return 1024 ** 2
    if "kib" in name:
        return 1024
    if "gb" in name:
        return 1_000_000_000
    if "mb" in name:
        return 1_000_000
    if "kb" in name:
        return 1000
    return 1.0


def parse_csv(raw: str) -> list[GPUReading]:
    """Parse raw ``nvidia-smi --format=csv`` text into :class:`GPUReading`\\ s.

    The header row is inspected to map columns by name (units in brackets are
    tolerated and used for byte normalisation), so the parser is robust to
    reordering and minor formatting differences across driver versions.

    Args:
        raw: Full stdout of ``nvidia-smi --query-gpu=... --format=csv``.

    Returns:
        A list with one :class:`GPUReading` per GPU. Empty if ``raw`` has no
        data rows.
    """
    lines = [line for line in raw.splitlines() if line.strip()]
    if not lines:
        return []

    header_cols = [col.strip() for col in lines[0].split(",")]
    # Map each canonical field to its column index using the keyword table.
    field_index: dict[str, Optional[int]] = {}
    for field_name, keywords in _FIELD_KEYWORDS.items():
        found: Optional[int] = None
        for idx, col in enumerate(header_cols):
            lowered = col.lower().replace(" ", "")
            if any(kw in lowered for kw in keywords):
                found = idx
                break
        field_index[field_name] = found

    # Pre-compute byte multipliers for byte-oriented columns.
    multipliers: dict[str, float] = {}
    for f in _BYTE_FIELDS:
        idx = field_index[f]
        if idx is not None and idx < len(header_cols):
            multipliers[f] = _unit_multiplier(header_cols[idx])
        else:
            multipliers[f] = 1.0

    def cell(field_name: str) -> str:
        idx = field_index[field_name]
        return parts[idx] if idx is not None and idx < len(parts) else ""

    def byte_value(field_name: str) -> Optional[int]:
        value = _clean_numeric(cell(field_name))
        if value is None:
            return None
        return int(value * multipliers[field_name])

    readings: list[GPUReading] = []
    for line in lines[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(header_cols):
            logger.warning("Skipping malformed CSV row: %r", line)
            continue
        index_raw = _clean_numeric(cell("index")) if field_index["index"] is not None else 0
        readings.append(
            GPUReading(
                index=int(index_raw if index_raw is not None else len(readings)),
                uuid=cell("uuid").strip() or f"gpu-{len(readings)}",
                name=cell("name").strip() or "unknown",
                utilization_percent=_clean_numeric(cell("utilization")) or 0.0,
                memory_used_bytes=byte_value("mem_used") or 0,
                memory_total_bytes=byte_value("mem_total") or 0,
                temperature_celsius=_clean_numeric(cell("temperature")) or 0.0,
                power_draw_watts=_clean_numeric(cell("power_draw")) or 0.0,
                power_limit_watts=_clean_numeric(cell("power_limit")) or 0.0,
                fan_speed_percent=_clean_numeric(cell("fan_speed")) or 0.0,
                pcie_tx_bytes=byte_value("pcie_tx"),
                pcie_rx_bytes=byte_value("pcie_rx"),
            )
        )
    return readings


def nvidia_smi_csv(binary: str = "nvidia-smi") -> str:
    """Run ``nvidia-smi`` and return its raw CSV stdout.

    Raises:
        FileNotFoundError: If the binary is not present on ``PATH``.
        RuntimeError: If ``nvidia-smi`` exits non-zero.
    """
    try:
        result = subprocess.run(
            [binary, f"--query-gpu={QUERY_FIELDS}", "--format=csv"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except FileNotFoundError as exc:  # binary missing entirely
        raise FileNotFoundError(f"{binary} not found on PATH") from exc

    if result.returncode != 0:
        raise RuntimeError(
            f"{binary} exited {result.returncode}: {result.stderr.strip()}"
        )
    return result.stdout


def collect(
    binary: str = "nvidia-smi", raw_text: Optional[str] = None
) -> tuple[list[GPUReading], bool]:
    """Collect GPU readings, degrading gracefully when no GPU is available.

    Args:
        binary: Name/path of the ``nvidia-smi`` binary for live collection.
        raw_text: When provided (mainly for tests), this raw CSV string is
            parsed instead of invoking the binary, and ``up`` is reported as
            ``True``.

    Returns:
        A tuple of ``(readings, up)`` where ``up`` is ``True`` when a successful
        collection happened and ``False`` when ``nvidia-smi`` was missing or
        errored (so downstream exporters can keep serving a ``0`` gauge).
    """
    if raw_text is not None:
        return parse_csv(raw_text), True

    try:
        raw = nvidia_smi_csv(binary)
        return parse_csv(raw), True
    except (FileNotFoundError, RuntimeError) as exc:
        logger.warning("GPU collection degraded: %s", exc)
        return [], False
