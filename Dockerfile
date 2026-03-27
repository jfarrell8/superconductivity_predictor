# ─── Stage 1: Builder ────────────────────────────────────────────────────────
# Install Python deps into a venv; this layer is cached across rebuilds
# as long as pyproject.toml hasn't changed.
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed by LightGBM / XGBoost C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
# Install only runtime deps (not dev) into an isolated venv
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install -e "."

# ─── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.source="https://github.com/jfarrell8/superconductivity-predictor"
LABEL org.opencontainers.image.description="Superconductivity Critical Temperature Predictor API"

# libgomp1 is required at runtime by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser
USER appuser
WORKDIR /app

# Copy venv from builder (no pip in runtime image)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source and required artifacts
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser configs/ ./configs/

# Model artifacts (populated by CI after training, or mounted via volume)
# COPY --chown=appuser:appuser models/ ./models/

# Expose FastAPI port
EXPOSE 8000

# Health check so orchestrators (k8s, ECS) know when the container is ready
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Gunicorn → uvicorn workers for production throughput
CMD ["python", "-m", "uvicorn", "src.api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
