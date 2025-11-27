#!/bin/sh

: << EOF
Setting up environment, and installing dependencies.
Designed to make sure that analysis can be completed without issues.
EOF

# Defining colours
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

set -e # exit on error

print_error() {
    echo -e "${RED}Error:${NC} $1"
}

print_success() {
    echo -e "${GREEN}Success:${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}Warning:${NC} $1"
}

# check whether uv is installed
echo "Checking whether uv is available"
if command -v uv &> /dev/null; then
    print_success "uv is already installed"
    uv --version
else
    print_warning "uv not found"
    echo "To properly use the backend of this repository, uv must be available in your system"
    if [ "$OS" == "macos" ]; then
        echo "On macos, to download uv using homebrew run the following command:"
        echo "      brew install uv"
        echo ""
    fi
    echo "To find out how to install uv, visit the following site:"
    echo "      https://docs.astral.sh/uv/"
    echo ""
    echo "You can also download and execute the installation shell script directly using the following commands:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "export PATH=\"$HOME/.cargo/bin:$PATH\""
    exit 1
fi
echo ""

# syncing packages
uv sync
