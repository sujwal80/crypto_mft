# ==============================================================================
# Dockerfile: NSE Intraday GEX-Micro System
# Base Image: Python 3.9-slim (optimized for size and speed)
# ==============================================================================

FROM python:3.9-slim

# Set environmental flags
# PYTHONUNBUFFERED=1 ensures real-time standard outputs are logged in Docker console
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Install required system compiler tools (scipy/numpy support)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --index-url https://pypi.org/simple

# Copy the application folder contents
# We copy the parent directory contents to mount packages cleanly
COPY . /app/indian_intraday_system

# Expose system logs/data directories
RUN mkdir -p /app/datasets /app/logs

# Create a non-privileged user 'quant' for container security best practices
RUN useradd -U quant && \
    chown -R quant:quant /app

# Switch to non-root user
USER quant

# Default command triggers GEX main orchestrator
CMD ["python", "indian_intraday_system/main.py"]
