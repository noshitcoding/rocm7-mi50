#!/usr/bin/env bash
# ============================================================================
# Quick install — Native Ollama on MI50 (gfx906) with ROCm 7.2
# ============================================================================
# Prerequisites:
#   - ROCm 7.2 installed at /opt/rocm
#   - Ollama installed (curl -fsSL https://ollama.com/install.sh | sh)
#   - cmake, git, build-essential
#
# Usage:
#   sudo ./install-native.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OLLAMA_VERSION="v0.17.7"

echo "=== Step 1: Create Tensile symlinks ==="
bash "${SCRIPT_DIR}/scripts/create-tensile-symlinks.sh"

echo ""
echo "=== Step 2: Build patched libggml-hip.so ==="
sudo -u "${SUDO_USER:-$(whoami)}" bash "${SCRIPT_DIR}/build.sh" "${OLLAMA_VERSION}"

echo ""
echo "=== Step 3: Install patched library ==="
BUILD_LIB="${SCRIPT_DIR}/build/ollama-src/build/lib/ollama/libggml-hip.so"
if [ -f "${BUILD_LIB}" ]; then
    cp "${BUILD_LIB}" /usr/local/lib/ollama/rocm/libggml-hip.so
    echo "Installed to /usr/local/lib/ollama/rocm/libggml-hip.so"
else
    echo "ERROR: Build artifact not found: ${BUILD_LIB}"
    exit 1
fi

echo ""
echo "=== Step 4: Install systemd override ==="
mkdir -p /etc/systemd/system/ollama.service.d/
cp "${SCRIPT_DIR}/ollama/override.conf" /etc/systemd/system/ollama.service.d/override.conf
systemctl daemon-reload

echo ""
echo "=== Step 5: Restart Ollama ==="
systemctl restart ollama
sleep 3

if systemctl is-active --quiet ollama; then
    echo "✓ Ollama is running!"
    echo ""
    echo "Test with:"
    echo "  ollama run llama3.2 'Hello!'"
    echo "  curl http://localhost:11434/api/generate -d '{\"model\":\"llama3.2\",\"prompt\":\"hi\",\"stream\":false}'"
else
    echo "✗ Ollama failed to start. Check: journalctl -u ollama -n 50"
    exit 1
fi
