#!/bin/bash
#
# Build .deb and .rpm packages for Garden Sentinel using fpm
#
# Prerequisites:
#   gem install fpm
#   apt-get install ruby ruby-dev build-essential rpm  # for Debian/Ubuntu
#   yum install ruby ruby-devel rpm-build              # for RHEL/CentOS
#

set -e

# Package metadata
VERSION="${VERSION:-0.1.0}"
ITERATION="${ITERATION:-1}"
MAINTAINER="Garden Sentinel <garden-sentinel@example.com>"
URL="https://github.com/brynmathias/garden-sentinel"
LICENSE="MIT"
VENDOR="Garden Sentinel"

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
DIST_DIR="$PROJECT_DIR/dist"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check for fpm
check_fpm() {
    if ! command -v fpm &>/dev/null; then
        log_error "fpm not found. Install with: gem install fpm"
        exit 1
    fi
    log_info "fpm version: $(fpm --version)"
}

# Clean build directories
clean() {
    log_info "Cleaning build directories..."
    rm -rf "$BUILD_DIR" "$DIST_DIR"
    mkdir -p "$BUILD_DIR" "$DIST_DIR"
}

# Prepare edge package files
prepare_edge() {
    log_info "Preparing edge package files..."

    local pkg_root="$BUILD_DIR/edge-root"
    mkdir -p "$pkg_root"

    # Application directory
    mkdir -p "$pkg_root/opt/garden-sentinel"
    cp -r "$PROJECT_DIR/garden_sentinel" "$pkg_root/opt/garden-sentinel/"
    cp -r "$PROJECT_DIR/config" "$pkg_root/opt/garden-sentinel/"

    # Config directory
    mkdir -p "$pkg_root/etc/garden-sentinel"
    cp "$PROJECT_DIR/config/edge.example.yaml" "$pkg_root/etc/garden-sentinel/edge.yaml.example"

    # Systemd service
    mkdir -p "$pkg_root/lib/systemd/system"
    cp "$PROJECT_DIR/systemd/garden-sentinel-edge.service" "$pkg_root/lib/systemd/system/"

    # Data directories (empty, will be created by postinst)
    mkdir -p "$pkg_root/var/lib/garden-sentinel"
    mkdir -p "$pkg_root/var/log/garden-sentinel"

    # Scripts directory
    mkdir -p "$pkg_root/opt/garden-sentinel/scripts"
    cp "$PROJECT_DIR/scripts/status.sh" "$pkg_root/opt/garden-sentinel/scripts/"

    # Python requirements
    cat > "$pkg_root/opt/garden-sentinel/requirements-edge.txt" << 'EOF'
numpy>=1.21.0
opencv-python-headless>=4.5.0
pillow>=9.0.0
pyyaml>=6.0
aiohttp>=3.8.0
websockets>=10.0
smbus2>=0.4.0
EOF

    echo "$pkg_root"
}

# Prepare server package files
prepare_server() {
    log_info "Preparing server package files..."

    local pkg_root="$BUILD_DIR/server-root"
    mkdir -p "$pkg_root"

    # Application directory
    mkdir -p "$pkg_root/opt/garden-sentinel"
    cp -r "$PROJECT_DIR/garden_sentinel" "$pkg_root/opt/garden-sentinel/"
    cp -r "$PROJECT_DIR/config" "$pkg_root/opt/garden-sentinel/"
    cp -r "$PROJECT_DIR/dashboards" "$pkg_root/opt/garden-sentinel/" 2>/dev/null || true

    # Config directory
    mkdir -p "$pkg_root/etc/garden-sentinel"
    cp "$PROJECT_DIR/config/server.example.yaml" "$pkg_root/etc/garden-sentinel/server.yaml.example"

    # Systemd service
    mkdir -p "$pkg_root/lib/systemd/system"
    cp "$PROJECT_DIR/systemd/garden-sentinel-server.service" "$pkg_root/lib/systemd/system/"

    # Data directories
    mkdir -p "$pkg_root/var/lib/garden-sentinel"
    mkdir -p "$pkg_root/var/log/garden-sentinel"

    # Scripts directory
    mkdir -p "$pkg_root/opt/garden-sentinel/scripts"
    cp "$PROJECT_DIR/scripts/status.sh" "$pkg_root/opt/garden-sentinel/scripts/"

    # Python requirements
    cat > "$pkg_root/opt/garden-sentinel/requirements-server.txt" << 'EOF'
numpy>=1.21.0
opencv-python-headless>=4.5.0
pillow>=9.0.0
pyyaml>=6.0
aiohttp>=3.8.0
websockets>=10.0
fastapi>=0.100.0
uvicorn>=0.22.0
paho-mqtt>=1.6.0
httpx>=0.24.0
torch>=2.0.0
ultralytics>=8.0.0
EOF

    echo "$pkg_root"
}

# Build edge packages
build_edge() {
    local pkg_root=$(prepare_edge)

    log_info "Building edge .deb package..."
    fpm \
        --input-type dir \
        --output-type deb \
        --name garden-sentinel-edge \
        --version "$VERSION" \
        --iteration "$ITERATION" \
        --architecture all \
        --maintainer "$MAINTAINER" \
        --vendor "$VENDOR" \
        --license "$LICENSE" \
        --url "$URL" \
        --description "Garden Sentinel Edge Device - AI-powered garden security camera" \
        --depends python3 \
        --depends python3-venv \
        --depends python3-pip \
        --depends i2c-tools \
        --config-files /etc/garden-sentinel/edge.yaml.example \
        --after-install "$PROJECT_DIR/packaging/postinst-edge.sh" \
        --before-remove "$PROJECT_DIR/packaging/prerm-edge.sh" \
        --after-remove "$PROJECT_DIR/packaging/postrm-edge.sh" \
        --chdir "$pkg_root" \
        --package "$DIST_DIR/garden-sentinel-edge_${VERSION}-${ITERATION}_all.deb" \
        .

    log_info "Building edge .rpm package..."
    fpm \
        --input-type dir \
        --output-type rpm \
        --name garden-sentinel-edge \
        --version "$VERSION" \
        --iteration "$ITERATION" \
        --architecture noarch \
        --maintainer "$MAINTAINER" \
        --vendor "$VENDOR" \
        --license "$LICENSE" \
        --url "$URL" \
        --description "Garden Sentinel Edge Device - AI-powered garden security camera" \
        --depends python3 \
        --depends i2c-tools \
        --config-files /etc/garden-sentinel/edge.yaml.example \
        --after-install "$PROJECT_DIR/packaging/postinst-edge.sh" \
        --before-remove "$PROJECT_DIR/packaging/prerm-edge.sh" \
        --after-remove "$PROJECT_DIR/packaging/postrm-edge.sh" \
        --chdir "$pkg_root" \
        --package "$DIST_DIR/garden-sentinel-edge-${VERSION}-${ITERATION}.noarch.rpm" \
        .

    log_info "Edge packages built successfully"
}

# Build server packages
build_server() {
    local pkg_root=$(prepare_server)

    log_info "Building server .deb package..."
    fpm \
        --input-type dir \
        --output-type deb \
        --name garden-sentinel-server \
        --version "$VERSION" \
        --iteration "$ITERATION" \
        --architecture all \
        --maintainer "$MAINTAINER" \
        --vendor "$VENDOR" \
        --license "$LICENSE" \
        --url "$URL" \
        --description "Garden Sentinel Server - Central detection and coordination server" \
        --depends python3 \
        --depends python3-venv \
        --depends python3-pip \
        --config-files /etc/garden-sentinel/server.yaml.example \
        --after-install "$PROJECT_DIR/packaging/postinst-server.sh" \
        --before-remove "$PROJECT_DIR/packaging/prerm-server.sh" \
        --after-remove "$PROJECT_DIR/packaging/postrm-server.sh" \
        --chdir "$pkg_root" \
        --package "$DIST_DIR/garden-sentinel-server_${VERSION}-${ITERATION}_all.deb" \
        .

    log_info "Building server .rpm package..."
    fpm \
        --input-type dir \
        --output-type rpm \
        --name garden-sentinel-server \
        --version "$VERSION" \
        --iteration "$ITERATION" \
        --architecture noarch \
        --maintainer "$MAINTAINER" \
        --vendor "$VENDOR" \
        --license "$LICENSE" \
        --url "$URL" \
        --description "Garden Sentinel Server - Central detection and coordination server" \
        --depends python3 \
        --config-files /etc/garden-sentinel/server.yaml.example \
        --after-install "$PROJECT_DIR/packaging/postinst-server.sh" \
        --before-remove "$PROJECT_DIR/packaging/prerm-server.sh" \
        --after-remove "$PROJECT_DIR/packaging/postrm-server.sh" \
        --chdir "$pkg_root" \
        --package "$DIST_DIR/garden-sentinel-server-${VERSION}-${ITERATION}.noarch.rpm" \
        .

    log_info "Server packages built successfully"
}

# Show usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] [COMMAND]

Commands:
    all         Build all packages (default)
    edge        Build edge device packages only
    server      Build server packages only
    clean       Clean build directories

Options:
    -v, --version VERSION    Set package version (default: $VERSION)
    -i, --iteration NUM      Set iteration/release number (default: $ITERATION)
    -h, --help               Show this help message

Examples:
    $0                      # Build all packages
    $0 -v 1.0.0 all         # Build all with version 1.0.0
    $0 edge                 # Build only edge packages
    $0 clean                # Clean build artifacts

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -i|--iteration)
            ITERATION="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        all|edge|server|clean)
            COMMAND="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Default command
COMMAND="${COMMAND:-all}"

# Main
main() {
    log_info "Garden Sentinel Package Builder"
    log_info "Version: $VERSION-$ITERATION"

    check_fpm

    case $COMMAND in
        all)
            clean
            build_edge
            build_server
            ;;
        edge)
            clean
            build_edge
            ;;
        server)
            clean
            build_server
            ;;
        clean)
            clean
            ;;
    esac

    log_info "Done! Packages are in: $DIST_DIR"
    ls -la "$DIST_DIR"
}

main
