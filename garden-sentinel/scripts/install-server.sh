#!/bin/bash
#
# Garden Sentinel Server Installation Script
# Run as root on the server machine
#

set -e

INSTALL_DIR="/opt/garden-sentinel"
CONFIG_DIR="/etc/garden-sentinel"
DATA_DIR="/var/lib/garden-sentinel"
LOG_DIR="/var/log/garden-sentinel"
SERVICE_USER="garden-sentinel"

echo "=== Garden Sentinel Server Installation ==="

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root (use sudo)"
   exit 1
fi

# Detect GPU
HAS_NVIDIA=false
if command -v nvidia-smi &>/dev/null; then
    HAS_NVIDIA=true
    echo "NVIDIA GPU detected"
fi

# Create service user
echo "Creating service user..."
if ! id "$SERVICE_USER" &>/dev/null; then
    useradd --system --no-create-home --shell /usr/sbin/nologin "$SERVICE_USER"
fi

# Add user to video/render groups for GPU access
usermod -aG video,render "$SERVICE_USER" 2>/dev/null || true

# Create directories
echo "Creating directories..."
mkdir -p "$INSTALL_DIR"
mkdir -p "$CONFIG_DIR"
mkdir -p "$DATA_DIR"/{frames,events,models}
mkdir -p "$LOG_DIR"

# Set ownership
chown -R "$SERVICE_USER:$SERVICE_USER" "$DATA_DIR"
chown -R "$SERVICE_USER:$SERVICE_USER" "$LOG_DIR"

# Install system dependencies
echo "Installing system dependencies..."
apt-get update
apt-get install -y \
    python3-pip \
    python3-venv \
    python3-opencv \
    libatlas-base-dev \
    libjpeg-dev \
    libopenblas-dev \
    git \
    curl

# Create Python virtual environment
echo "Creating Python virtual environment..."
python3 -m venv "$INSTALL_DIR/venv"
source "$INSTALL_DIR/venv/bin/activate"

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip wheel

# Install PyTorch (with CUDA if available)
if [[ "$HAS_NVIDIA" == true ]]; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
    echo "Installing PyTorch (CPU only)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# Install server dependencies
pip install \
    numpy \
    opencv-python-headless \
    pillow \
    pyyaml \
    aiohttp \
    websockets \
    fastapi \
    uvicorn \
    paho-mqtt \
    httpx \
    ultralytics

# Copy application files
echo "Copying application files..."
if [[ -d "./garden_sentinel" ]]; then
    cp -r ./garden_sentinel "$INSTALL_DIR/"
else
    echo "Warning: garden_sentinel directory not found in current directory"
    echo "Please copy the application files manually to $INSTALL_DIR/garden_sentinel"
fi

# Copy dashboards
if [[ -d "./dashboards" ]]; then
    cp -r ./dashboards "$INSTALL_DIR/"
fi

# Install the package
pip install -e "$INSTALL_DIR" 2>/dev/null || true

# Copy example config if no config exists
if [[ ! -f "$CONFIG_DIR/server.yaml" ]]; then
    echo "Installing example configuration..."
    if [[ -f "./config/server.example.yaml" ]]; then
        cp ./config/server.example.yaml "$CONFIG_DIR/server.yaml"
    elif [[ -f "$INSTALL_DIR/config/server.example.yaml" ]]; then
        cp "$INSTALL_DIR/config/server.example.yaml" "$CONFIG_DIR/server.yaml"
    fi
    echo "Please edit $CONFIG_DIR/server.yaml with your settings"
fi

# Install systemd service
echo "Installing systemd service..."
cp ./systemd/garden-sentinel-server.service /etc/systemd/system/
systemctl daemon-reload

# Enable service
echo "Enabling service..."
systemctl enable garden-sentinel-server.service

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit configuration: sudo nano $CONFIG_DIR/server.yaml"
echo "     - Configure your cameras"
echo "     - Set up spatial zones for your garden"
echo "     - Configure notifications (push, email, etc.)"
echo ""
echo "  2. Download/train a model and place in $DATA_DIR/models/"
echo ""
echo "  3. Start the service: sudo systemctl start garden-sentinel-server"
echo ""
echo "  4. Check status: sudo systemctl status garden-sentinel-server"
echo ""
echo "  5. View logs: sudo journalctl -u garden-sentinel-server -f"
echo ""
echo "  6. Access the web UI at http://localhost:5000"
echo ""
echo "Prometheus metrics available at http://localhost:9090/metrics"
echo ""
if [[ "$HAS_NVIDIA" == true ]]; then
    echo "GPU: NVIDIA GPU detected and CUDA support installed"
else
    echo "GPU: No NVIDIA GPU detected, using CPU inference"
fi
