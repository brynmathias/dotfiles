#!/bin/bash
#
# Garden Sentinel Edge Device Installation Script
# Run as root on Raspberry Pi
#

set -e

INSTALL_DIR="/opt/garden-sentinel"
CONFIG_DIR="/etc/garden-sentinel"
DATA_DIR="/var/lib/garden-sentinel"
LOG_DIR="/var/log/garden-sentinel"
SERVICE_USER="garden-sentinel"

echo "=== Garden Sentinel Edge Device Installation ==="

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root (use sudo)"
   exit 1
fi

# Detect architecture
ARCH=$(uname -m)
echo "Detected architecture: $ARCH"

# Create service user
echo "Creating service user..."
if ! id "$SERVICE_USER" &>/dev/null; then
    useradd --system --no-create-home --shell /usr/sbin/nologin "$SERVICE_USER"
fi

# Add user to hardware groups
usermod -aG video,gpio,i2c,spi "$SERVICE_USER" 2>/dev/null || true

# Create directories
echo "Creating directories..."
mkdir -p "$INSTALL_DIR"
mkdir -p "$CONFIG_DIR"
mkdir -p "$DATA_DIR"/{offline,recordings}
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
    i2c-tools \
    git

# Enable I2C and SPI
echo "Enabling I2C and SPI..."
if ! grep -q "^dtparam=i2c_arm=on" /boot/config.txt 2>/dev/null; then
    echo "dtparam=i2c_arm=on" >> /boot/config.txt
fi
if ! grep -q "^dtparam=spi=on" /boot/config.txt 2>/dev/null; then
    echo "dtparam=spi=on" >> /boot/config.txt
fi

# Load I2C module
modprobe i2c-dev 2>/dev/null || true
if ! grep -q "^i2c-dev" /etc/modules; then
    echo "i2c-dev" >> /etc/modules
fi

# Create Python virtual environment
echo "Creating Python virtual environment..."
python3 -m venv "$INSTALL_DIR/venv"
source "$INSTALL_DIR/venv/bin/activate"

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip wheel

# Install edge-specific packages
pip install \
    numpy \
    opencv-python-headless \
    pillow \
    pyyaml \
    aiohttp \
    websockets \
    smbus2 \
    RPi.GPIO \
    picamera2

# Install Coral TPU support (if available)
if [[ "$ARCH" == "aarch64" ]]; then
    echo "Installing Coral TPU support..."
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | \
        tee /etc/apt/sources.list.d/coral-edgetpu.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
    apt-get update
    apt-get install -y libedgetpu1-std python3-pycoral || echo "Coral packages not available"
    pip install tflite-runtime || true
fi

# Copy application files
echo "Copying application files..."
if [[ -d "./garden_sentinel" ]]; then
    cp -r ./garden_sentinel "$INSTALL_DIR/"
else
    echo "Warning: garden_sentinel directory not found in current directory"
    echo "Please copy the application files manually to $INSTALL_DIR/garden_sentinel"
fi

# Install the package
pip install -e "$INSTALL_DIR" 2>/dev/null || true

# Copy example config if no config exists
if [[ ! -f "$CONFIG_DIR/edge.yaml" ]]; then
    echo "Installing example configuration..."
    if [[ -f "./config/edge.example.yaml" ]]; then
        cp ./config/edge.example.yaml "$CONFIG_DIR/edge.yaml"
    elif [[ -f "$INSTALL_DIR/config/edge.example.yaml" ]]; then
        cp "$INSTALL_DIR/config/edge.example.yaml" "$CONFIG_DIR/edge.yaml"
    fi
    echo "Please edit $CONFIG_DIR/edge.yaml with your settings"
fi

# Install systemd service
echo "Installing systemd service..."
cp ./systemd/garden-sentinel-edge.service /etc/systemd/system/
systemctl daemon-reload

# Enable and start service
echo "Enabling service..."
systemctl enable garden-sentinel-edge.service

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit configuration: sudo nano $CONFIG_DIR/edge.yaml"
echo "     - Set your device ID"
echo "     - Configure server address"
echo "     - Set GPIO pins for your hardware"
echo ""
echo "  2. Start the service: sudo systemctl start garden-sentinel-edge"
echo ""
echo "  3. Check status: sudo systemctl status garden-sentinel-edge"
echo ""
echo "  4. View logs: sudo journalctl -u garden-sentinel-edge -f"
echo ""
echo "Note: A reboot may be required for I2C/SPI changes to take effect"
