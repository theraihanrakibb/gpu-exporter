# syntax=docker/dockerfile:1

# gpu-exporter: Prometheus exporter for NVIDIA GPU metrics.
# Runtime note: this image only *serves* metrics. To see real GPU values the
# container must have access to `nvidia-smi` (e.g. run with the NVIDIA
# Container Toolkit: `--gpus all`). Without it the exporter still runs and
# exposes `gpu_collector_up 0` so it degrades gracefully on CPU-only hosts.
FROM python:3.11-slim

# Run as a non-root user.
RUN groupadd --system app && useradd --system --gid app --create-home app

WORKDIR /app

# Install dependencies first to leverage Docker layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the package and tests.
COPY gpu_exporter ./gpu_exporter

# The exporter listens on this port by default (override with --port).
ENV PORT=9400
EXPOSE 9400

USER app

# Serve GPU metrics. --collector-text can be supplied for offline/CI runs.
CMD ["python", "-m", "gpu_exporter", "--port", "9400"]
