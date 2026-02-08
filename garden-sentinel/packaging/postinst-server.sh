#!/bin/bash
#
# Post-installation script for garden-sentinel-server
#

set -e

INSTALL_DIR="/opt/garden-sentinel"
CONFIG_DIR="/etc/garden-sentinel"
DATA_DIR="/var/lib/garden-sentinel"
LOG_DIR="/var/log/garden-sentinel"
SERVICE_USER="garden-sentinel"

echo "Configuring Garden Sentinel Server..."

# Create service user if not exists
if ! id "$SERVICE_USER" &>/dev/null; then
    echo "Creating service user: $SERVICE_USER"
    useradd --system --no-create-home --shell /usr/sbin/nologin "$SERVICE_USER"
fi

# Add user to video/render groups for GPU access
for group in video render; do
    if getent group "$group" &>/dev/null; then
        usermod -aG "$group" "$SERVICE_USER" 2>/dev/null || true
    fi
done

# Create data directories
mkdir -p "$DATA_DIR"/{frames,events,models}
mkdir -p "$LOG_DIR"

# Set ownership
chown -R "$SERVICE_USER:$SERVICE_USER" "$DATA_DIR"
chown -R "$SERVICE_USER:$SERVICE_USER" "$LOG_DIR"

# Create Python virtual environment
if [ ! -d "$INSTALL_DIR/venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$INSTALL_DIR/venv"
fi

# Detect GPU
HAS_NVIDIA=false
if command -v nvidia-smi &>/dev/null; then
    HAS_NVIDIA=true
    echo "NVIDIA GPU detected"
fi

# Install Python dependencies
echo "Installing Python dependencies..."
"$INSTALL_DIR/venv/bin/pip" install --upgrade pip wheel

# Install base requirements (without torch - handle separately)
"$INSTALL_DIR/venv/bin/pip" install \
    numpy \
    opencv-python-headless \
    pillow \
    pyyaml \
    aiohttp \
    websockets \
    fastapi \
    uvicorn \
    paho-mqtt \
    httpx

# Install PyTorch with appropriate backend
if [ "$HAS_NVIDIA" = true ]; then
    echo "Installing PyTorch with CUDA support..."
    "$INSTALL_DIR/venv/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
    echo "Installing PyTorch (CPU only)..."
    "$INSTALL_DIR/venv/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# Install ultralytics (YOLO)
"$INSTALL_DIR/venv/bin/pip" install ultralytics

# Copy example config if no config exists
if [ ! -f "$CONFIG_DIR/server.yaml" ]; then
    if [ -f "$CONFIG_DIR/server.yaml.example" ]; then
        cp "$CONFIG_DIR/server.yaml.example" "$CONFIG_DIR/server.yaml"
        echo "Created default config: $CONFIG_DIR/server.yaml"
        echo "Please edit this file with your settings"
    fi
fi

# Reload systemd
systemctl daemon-reload

echo ""
echo "============================================"
echo "Garden Sentinel Server installation complete"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Edit config: sudo nano $CONFIG_DIR/server.yaml"
echo "  2. Place model in: $DATA_DIR/models/"
echo "  3. Enable service: sudo systemctl enable garden-sentinel-server"
echo "  4. Start service: sudo systemctl start garden-sentinel-server"
echo "  5. View logs: sudo journalctl -u garden-sentinel-server -f"
echo ""
echo "Web UI will be available at: http://localhost:5000"
echo "Prometheus metrics at: http://localhost:9090/metrics"
echo ""
if [ "$HAS_NVIDIA" = true ]; then
    echo "GPU: NVIDIA GPU detected, CUDA support installed"
else
    echo "GPU: No NVIDIA GPU detected, using CPU inference"
fi
echo ""

exit 0
