#!/bin/bash
#
# Pre-removal script for garden-sentinel-server
#

set -e

echo "Stopping Garden Sentinel Server service..."

# Stop the service if running
if systemctl is-active --quiet garden-sentinel-server 2>/dev/null; then
    systemctl stop garden-sentinel-server
fi

# Disable the service
if systemctl is-enabled --quiet garden-sentinel-server 2>/dev/null; then
    systemctl disable garden-sentinel-server
fi

echo "Service stopped and disabled"

exit 0
