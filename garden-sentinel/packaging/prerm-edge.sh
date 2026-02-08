#!/bin/bash
#
# Pre-removal script for garden-sentinel-edge
#

set -e

echo "Stopping Garden Sentinel Edge service..."

# Stop the service if running
if systemctl is-active --quiet garden-sentinel-edge 2>/dev/null; then
    systemctl stop garden-sentinel-edge
fi

# Disable the service
if systemctl is-enabled --quiet garden-sentinel-edge 2>/dev/null; then
    systemctl disable garden-sentinel-edge
fi

echo "Service stopped and disabled"

exit 0
