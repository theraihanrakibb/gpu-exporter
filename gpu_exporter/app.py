"""HTTP exposer that serves GPU metrics for Prometheus scraping.

The :class:`Exporter` starts ``prometheus_client``'s HTTP server and runs a
background collection loop. It is usable both as a long-running service (via the
CLI) and embedded in another process (as a library).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from prometheus_client import CollectorRegistry, start_http_server

from . import collector
from .metrics import MetricsRegistry

logger = logging.getLogger(__name__)

DEFAULT_PORT = 9400
DEFAULT_INTERVAL = 5.0


class Exporter:
    """Collects GPU readings on an interval and serves them over HTTP."""

    def __init__(
        self,
        port: int = DEFAULT_PORT,
        interval: float = DEFAULT_INTERVAL,
        collector_text: Optional[str] = None,
        binary: str = "nvidia-smi",
        registry: Optional[CollectorRegistry] = None,
    ) -> None:
        """Configure the exporter.

        Args:
            port: TCP port for the ``/metrics`` endpoint.
            interval: Seconds between collections.
            collector_text: Raw nvidia-smi CSV to parse instead of running the
                binary (handy for testing or static fixtures).
            binary: Path/name of ``nvidia-smi`` for live collection.
            registry: Prometheus registry to register gauges on.
        """
        self.port = port
        self.interval = interval
        self.collector_text = collector_text
        self.binary = binary
        self.metrics = MetricsRegistry(registry)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _tick(self) -> None:
        readings, up = collector.collect(
            binary=self.binary, raw_text=self.collector_text
        )
        self.metrics.update(readings, up)
        if up:
            logger.info("Collected %d GPU(s)", len(readings))
        else:
            logger.warning("Collector reported down; serving degraded metrics")

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            self._tick()
            self._stop.wait(self.interval)

    def start(self) -> None:
        """Start the HTTP server and the background collection loop."""
        start_http_server(self.port, registry=self.metrics.registry)
        self._tick()  # populate immediately so /metrics is non-empty
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("gpu-exporter serving /metrics on port %d", self.port)

    def stop(self) -> None:
        """Signal the collection loop to stop."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
