#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# setup_oracle.sh — One-shot setup for Oracle Cloud always-free ARM instance
#
# Prerequisites:
#   1. Create an Oracle Cloud account (free tier)
#   2. Create an A1.Flex VM instance:
#      - Image: Ubuntu 22.04 (or 24.04)
#      - Shape: VM.Standard.A1.Flex (always free: 4 OCPUs, 24 GB RAM)
#      - Upload your SSH public key
#   3. Open port 22 in the security list (for SSH)
#   4. SSH into the instance, then run this script
#
# Usage:
#   chmod +x setup_oracle.sh
#   ./setup_oracle.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

echo "=== Greenhouse Optimizer — Oracle Cloud Setup ==="
echo ""

# ── 1. System packages ─────────────────────────────────────────────────────
echo "[1/5] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq docker.io docker-compose-plugin git

sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker "$USER"
echo "  ✓ Docker installed"

# ── 2. Clone repo ──────────────────────────────────────────────────────────
echo ""
echo "[2/5] Cloning greenhouse repo..."
REPO_DIR="$HOME/greenhouse_digital_twin"
if [ -d "$REPO_DIR" ]; then
    echo "  Repo already exists at $REPO_DIR, pulling latest..."
    cd "$REPO_DIR" && git pull
else
    # Update this URL if your repo is on GitHub
    git clone https://github.com/YOUR_USERNAME/greenhouse_digital_twin.git "$REPO_DIR"
    cd "$REPO_DIR"
fi
echo "  ✓ Repo ready at $REPO_DIR"

# ── 3. Create .env file ────────────────────────────────────────────────────
echo ""
echo "[3/5] Creating .env file..."
ENV_FILE="$REPO_DIR/finished_optimization_code/.env"
if [ ! -f "$ENV_FILE" ]; then
    cat > "$ENV_FILE" << 'ENVEOF'
# Greenhouse Optimizer — Environment Variables
# Edit these values, then restart the daemon:
#   docker compose up -d --build

GREENHOUSE_ENDPOINT_URL=https://greenhouse-api.ffnfghnhzt.workers.dev/
GREENHOUSE_ENDPOINT_PASSWORD=s1717
GREENHOUSE_INTERVAL=900
GREENHOUSE_SOC_INIT=50
GREENHOUSE_PUMP_LOCKOUT=false
GREENHOUSE_QUIET=false
ENVEOF
    echo "  ✓ Created $ENV_FILE"
    echo "  ⚠ EDIT THIS FILE with your settings before starting the daemon!"
else
    echo "  ✓ .env already exists, skipping"
fi

# ── 4. Build and start ────────────────────────────────────────────────────
echo ""
echo "[4/5] Building Docker image..."
cd "$REPO_DIR/finished_optimization_code"
docker compose build

echo ""
echo "[5/5] Starting daemon..."
docker compose up -d

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Commands:"
echo "  docker compose logs -f          # watch live logs"
echo "  docker compose down             # stop the daemon"
echo "  docker compose up -d --build    # rebuild and restart"
echo "  docker compose restart          # restart without rebuild"
echo ""
echo "The daemon runs every 15 minutes (configurable in .env)."
echo "Edit $ENV_FILE, then: docker compose up -d --build"
