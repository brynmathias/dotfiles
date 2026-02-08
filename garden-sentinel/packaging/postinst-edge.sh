#!/bin/bash
#
# Post-installation script for garden-sentinel-edge
#

set -e

INSTALL_DIR="/opt/garden-sentinel"
CONFIG_DIR="/etc/garden-sentinel"
DATA_DIR="/var/lib/garden-sentinel"
LOG_DIR="/var/log/garden-sentinel"
SERVICE_USER="garden-sentinel"

echo "Configuring Garden Sentinel Edge..."

# Create service user if not exists
if ! id "$SERVICE_USER" &>/dev/null; then
    echo "Creating service user: $SERVICE_USER"
    useradd --system --no-create-home --shell /usr/sbin/nologin "$SERVICE_USER"
fi

# Add user to hardware groups (Raspberry Pi specific)
for group in video gpio i2c spi; do
    if getent group "$group" &>/dev/null; then
        usermod -aG "$group" "$SERVICE_USER" 2>/dev/null || true
    fi
done

# Create data directories
mkdir -p "$DATA_DIR"/{offline,recordings}
mkdir -p "$LOG_DIR"

# Set ownership
chown -R "$SERVICE_USER:$SERVICE_USER" "$DATA_DIR"
chown -R "$SERVICE_USER:$SERVICE_USER" "$LOG_DIR"

# Create Python virtual environment
if [ ! -d "$INSTALL_DIR/venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$INSTALL_DIR/venv"
fi

# Install Python dependencies
echo "Installing Python dependencies..."
"$INSTALL_DIR/venv/bin/pip" install --upgrade pip wheel
"$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements-edge.txt"

# Try to install Raspberry Pi specific packages
"$INSTALL_DIR/venv/bin/pip" install RPi.GPIO 2>/dev/null || echo "Note: RPi.GPIO not installed (not on Pi?)"
"$INSTALL_DIR/venv/bin/pip" install picamera2 2>/dev/null || echo "Note: picamera2 not installed (not on Pi?)"

# Try to install Coral TPU support
"$INSTALL_DIR/venv/bin/pip" install tflite-runtime 2>/dev/null || echo "Note: tflite-runtime not installed"

# Copy example config if no config exists
if [ ! -f "$CONFIG_DIR/edge.yaml" ]; then
    if [ -f "$CONFIG_DIR/edge.yaml.example" ]; then
        cp "$CONFIG_DIR/edge.yaml.example" "$CONFIG_DIR/edge.yaml"
        echo "Created default config: $CONFIG_DIR/edge.yaml"
        echo "Please edit this file with your settings"
    fi
fi

# Enable I2C if on Raspberry Pi
if [ -f /boot/config.txt ]; then
    if ! grep -q "^dtparam=i2c_arm=on" /boot/config.txt; then
        echo "dtparam=i2c_arm=on" >> /boot/config.txt
        echo "Enabled I2C in /boot/config.txt (reboot required)"
    fi
fi

# Load I2C module
modprobe i2c-dev 2>/dev/null || true
if [ -f /etc/modules ] && ! grep -q "^i2c-dev" /etc/modules; then
    echo "i2c-dev" >> /etc/modules
fi

# Reload systemd
systemctl daemon-reload

echo ""
echo "=========================================="
echo "Garden Sentinel Edge installation complete"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Edit config: sudo nano $CONFIG_DIR/edge.yaml"
echo "  2. Enable service: sudo systemctl enable garden-sentinel-edge"
echo "  3. Start service: sudo systemctl start garden-sentinel-edge"
echo "  4. View logs: sudo journalctl -u garden-sentinel-edge -f"
echo ""

exit 0
