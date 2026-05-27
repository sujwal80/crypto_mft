# Use the official lightweight Python base image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/crypto_swing_trading

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt (from parent build context)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire crypto_swing_trading application into the container
COPY crypto_swing_trading/ ./crypto_swing_trading/

# Create folders for logs and reports
RUN mkdir -p /app/logs /app/reports

# Expose default environment variables for runtime overrides
ENV TRADING_MODE=SHADOW
ENV SYMBOL=BTCUSDT

# Default execution command runs the Master Orchestrator
CMD ["python", "crypto_swing_trading/main.py"]
