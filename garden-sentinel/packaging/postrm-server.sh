#!/bin/bash
#
# Post-removal script for garden-sentinel-server
#

# Only run on purge (not upgrade)
if [ "$1" = "purge" ] || [ "$1" = "0" ]; then
    echo "Cleaning up Garden Sentinel Server..."

    # Remove virtual environment
    rm -rf /opt/garden-sentinel/venv

    # Optionally remove data (commented out for safety)
    # rm -rf /var/lib/garden-sentinel
    # rm -rf /var/log/garden-sentinel

    # Remove config only on purge
    rm -f /etc/garden-sentinel/server.yaml

    # Remove user if no other garden-sentinel packages installed
    if ! dpkg -l | grep -q "garden-sentinel-edge"; then
        if id garden-sentinel &>/dev/null; then
            echo "Removing service user..."
            userdel garden-sentinel 2>/dev/null || true
        fi
    fi

    echo "Cleanup complete"
    echo ""
    echo "Note: Data in /var/lib/garden-sentinel was preserved"
    echo "Remove manually if no longer needed"
fi

# Reload systemd
systemctl daemon-reload 2>/dev/null || true

exit 0
