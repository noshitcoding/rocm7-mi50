#!/usr/bin/env bash
# ============================================================================
# Build patched Ollama libggml-hip.so for MI50 (gfx906) on ROCm 7.2
# ============================================================================
# Usage (native build, no Docker):
#   ./build.sh [OLLAMA_VERSION]
#
# Example:
#   ./build.sh v0.17.7
#
# Prerequisites:
#   - ROCm 7.2 installed at /opt/rocm
#   - cmake, git, build-essential
# ============================================================================
set -euo pipefail

OLLAMA_VERSION="${1:-v0.17.7}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
SRC_DIR="${BUILD_DIR}/ollama-src"

echo "==> Building Ollama ${OLLAMA_VERSION} libggml-hip.so for gfx906..."

# Clone source
if [ ! -d "${SRC_DIR}" ]; then
    echo "==> Cloning Ollama ${OLLAMA_VERSION}..."
    git clone --depth 1 --branch "${OLLAMA_VERSION}" \
        https://github.com/ollama/ollama.git "${SRC_DIR}"
fi

# Apply patches
echo "==> Applying gfx906 patches..."
cd "${SRC_DIR}"
git checkout -- . 2>/dev/null || true
git apply "${SCRIPT_DIR}/patches/gfx906-rocm72-all.diff"

# Build
echo "==> Configuring CMake..."
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DAMDGPU_TARGETS="gfx906" \
    -DGGML_CUDA_FORCE_MMQ=ON \
    -DGGML_HIP_NO_MMQ_MFMA=ON \
    -DCMAKE_PREFIX_PATH="/opt/rocm"

echo "==> Building libggml-hip.so..."
cmake --build build --target ggml-hip -j"$(nproc)"

echo ""
echo "==> Build complete!"
echo "    Library: ${SRC_DIR}/build/lib/ollama/libggml-hip.so"
echo ""
echo "==> To install (native, non-Docker):"
echo "    sudo cp ${SRC_DIR}/build/lib/ollama/libggml-hip.so /usr/local/lib/ollama/rocm/"
echo "    sudo systemctl restart ollama"
