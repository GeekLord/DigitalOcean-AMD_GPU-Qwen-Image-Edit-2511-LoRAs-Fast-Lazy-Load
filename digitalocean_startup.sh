#!/bin/bash
# ==============================================================================
# DigitalOcean Droplet Cloud-Init User Data / Startup Script
# Target Hardware: AMD Instinct MI300X GPU (ROCm)
# Purpose: Auto-install ROCm dependencies, clone repo, and start Qwen Image Edit service
# ==============================================================================

set -euo pipefail

LOG_FILE="/var/log/qwen-startup.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== [1/6] Starting DigitalOcean MI300X Setup: $(date) ==="

# ------------------------------------------------------------------------------
# 1. Update system packages & install base dependencies
# ------------------------------------------------------------------------------
echo "=== [2/6] Updating System & Installing Build Utilities ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    python3-pip \
    python3-venv \
    python3-dev \
    ufw \
    ca-certificates

# ------------------------------------------------------------------------------
# 2. Configure Firewall (UFW)
# ------------------------------------------------------------------------------
echo "=== [3/6] Configuring Firewall for Port 7860 (Gradio Web UI) ==="
ufw allow 22/tcp
ufw allow 7860/tcp
ufw --force enable || true

# ------------------------------------------------------------------------------
# 3. Setup NVMe Scratch Cache Directory
# ------------------------------------------------------------------------------
echo "=== [4/6] Setting up Scratch NVMe Disk Cache ==="
SCRATCH_DIR="/mnt/scratch/hf_cache"
mkdir -p "${SCRATCH_DIR}"
chmod 777 "${SCRATCH_DIR}"

# ------------------------------------------------------------------------------
# 4. Clone Repository & Setup Virtual Environment
# ------------------------------------------------------------------------------
echo "=== [5/6] Deploying Qwen-Image-Edit Project & Installing ROCm PyTorch ==="
INSTALL_DIR="/opt/Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load"
REPO_URL="https://github.com/GeekLord/DigitalOcean-AMD_GPU-Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load.git"

if [ -d "${INSTALL_DIR}" ]; then
    echo "Directory ${INSTALL_DIR} exists, pulling latest..."
    cd "${INSTALL_DIR}"
    git pull || true
else
    git clone "${REPO_URL}" "${INSTALL_DIR}"
    cd "${INSTALL_DIR}"
fi

# Create venv and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements-rocm.txt

# ------------------------------------------------------------------------------
# 5. Create Systemd Service for Auto-Start
# ------------------------------------------------------------------------------
echo "=== [6/6] Creating Systemd Service 'qwen-image-edit' ==="

SERVICE_FILE="/etc/systemd/system/qwen-image-edit.service"
cat << 'EOF' > "${SERVICE_FILE}"
[Unit]
Description=Qwen Image Edit 2511 LoRAs Web Service (AMD MI300X)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load
Environment="HF_HOME=/mnt/scratch/hf_cache"
Environment="GRADIO_SERVER_NAME=0.0.0.0"
Environment="GRADIO_SERVER_PORT=7860"
ExecStart=/opt/Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load/.venv/bin/python app.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable qwen-image-edit.service
systemctl start qwen-image-edit.service

echo "=== Setup Complete! Qwen Image Edit service is active at http://<DROPLET_PUBLIC_IP>:7860/ ==="
