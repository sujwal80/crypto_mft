#!/usr/bin/env bash
# ==============================================================================
# deploy.sh: Automated Docker Deployment Rig for AWS Mumbai (ap-south-1)
# ==============================================================================

set -e

echo "=============================================================================="
# Step 1: Verify Docker and Docker Compose are installed
if ! [ -x "$(command -v docker)" ]; then
  echo 'Error: Docker is not installed. Please install Docker on your EC2 instance.' >&2
  exit 1
fi

if ! [ -x "$(command -v docker-compose)" ] && ! docker compose version &>/dev/null; then
  echo 'Error: Docker Compose is not installed. Please install Docker Compose.' >&2
  exit 1
fi

# Step 2: Set file permissions on system environments
if [ ! -f .env ]; then
  echo "[Warning] Local .env configuration profile file not found!"
  echo "Copying .env.example or creating default..."
  echo "SYSTEM_ENVIRONMENT=\"PAPER\"" >> .env
  echo "BROKER_ROUTER_CLASS=\"SHADOW_ROUTER\"" >> .env
  echo "TRUEDATA_USERNAME=\"placeholder_user\"" >> .env
  echo "TRUEDATA_PASSWORD=\"placeholder_pass\"" >> .env
  echo "TRUEDATA_WS_URL=\"wss://api.truedata.in/v3/websocket\"" >> .env
fi

# Step 3: Build and spin up Docker containers in detached (background) mode
echo "[Deploy] Building and spinning up containers in background..."
# Supports both 'docker-compose' and newer 'docker compose' CLI extensions
if docker compose version &>/dev/null; then
  docker compose up -d --build
else
  docker-compose up -d --build
fi

echo "=============================================================================="
echo "              indian_intraday_system DEPLOYMENT COMPLETED SUCCESSFULLY                "
echo "=============================================================================="
echo "Check live logs:    docker logs -f nse_gex_bot"
echo "Check active state:   docker ps -a | grep nse_gex_bot"
echo "Emergency halt:       docker stop nse_gex_bot"
echo "=============================================================================="
