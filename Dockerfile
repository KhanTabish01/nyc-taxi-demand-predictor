# Multi-stage build for NYC Taxi Demand Predictor API

# Stage 1: Builder
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata files needed by flit_core
COPY pyproject.toml README.md LICENSE ./
COPY requirements.txt .
COPY urban_mobility_forecaster/ ./urban_mobility_forecaster/

# Install Python dependencies to system site-packages (not --user)
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy all site-packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code and models
COPY urban_mobility_forecaster/ ./urban_mobility_forecaster/
COPY models/ ./models/

# Create non-root user for security
RUN useradd -m -u 1000 predictor && \
    chown -R predictor:predictor /app

USER predictor

# Health check (using urllib to avoid extra dependency)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://localhost:8000/health').read()"

# Expose port
EXPOSE 8000

# Run API server
CMD ["python", "-m", "uvicorn", "urban_mobility_forecaster.api:app", \
     "--host", "0.0.0.0", "--port", "8000"]
