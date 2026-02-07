#!/bin/bash
#
# Garden Sentinel Status Check
#

echo "=== Garden Sentinel Status ==="
echo ""

# Check if edge service exists and get status
if systemctl list-unit-files | grep -q garden-sentinel-edge; then
    echo "Edge Service:"
    systemctl status garden-sentinel-edge --no-pager -l 2>/dev/null || echo "  Not running"
    echo ""
fi

# Check if server service exists and get status
if systemctl list-unit-files | grep -q garden-sentinel-server; then
    echo "Server Service:"
    systemctl status garden-sentinel-server --no-pager -l 2>/dev/null || echo "  Not running"
    echo ""
fi

# Check ports
echo "Listening Ports:"
ss -tlnp 2>/dev/null | grep -E ':(5000|8080|9090|9091)' || echo "  No Garden Sentinel ports found"
echo ""

# Check disk usage
echo "Disk Usage:"
if [[ -d /var/lib/garden-sentinel ]]; then
    du -sh /var/lib/garden-sentinel/* 2>/dev/null || echo "  Data directory empty"
else
    echo "  Data directory not found"
fi
echo ""

# Check recent logs
echo "Recent Logs (last 10 lines):"
if systemctl list-unit-files | grep -q garden-sentinel-edge; then
    echo "--- Edge ---"
    journalctl -u garden-sentinel-edge -n 10 --no-pager 2>/dev/null || echo "  No logs"
fi
if systemctl list-unit-files | grep -q garden-sentinel-server; then
    echo "--- Server ---"
    journalctl -u garden-sentinel-server -n 10 --no-pager 2>/dev/null || echo "  No logs"
fi
