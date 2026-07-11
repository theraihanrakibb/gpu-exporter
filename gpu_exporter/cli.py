"""Command-line entry point for the GPU exporter."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Optional, Sequence

from .app import DEFAULT_INTERVAL, DEFAULT_PORT, Exporter

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="gpu-exporter",
        description="Prometheus exporter for NVIDIA GPU metrics (nvidia-smi).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to serve /metrics on (default: {DEFAULT_PORT}).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL,
        help=f"Collection interval in seconds (default: {DEFAULT_INTERVAL}).",
    )
    parser.add_argument(
        "--collector-text",
        default=None,
        help="Raw nvidia-smi CSV to parse instead of running the binary "
        "(useful for testing and CI).",
    )
    parser.add_argument(
        "--binary",
        default="nvidia-smi",
        help="Path or name of the nvidia-smi binary (default: nvidia-smi).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: INFO).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse arguments, start the exporter, and block until interrupted."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format=_LOG_FORMAT)
    logging.getLogger("gpu_exporter").setLevel(getattr(logging, args.log_level))

    exporter = Exporter(
        port=args.port,
        interval=args.interval,
        collector_text=args.collector_text,
        binary=args.binary,
    )
    try:
        exporter.start()
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Shutting down gpu-exporter")
        exporter.stop()
        return 0


if __name__ == "__main__":
    sys.exit(main())
