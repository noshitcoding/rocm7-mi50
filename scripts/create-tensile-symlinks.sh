#!/usr/bin/env bash
# ============================================================================
# Create gfx906 -> gfx908 Tensile library symlinks
# ============================================================================
# ROCm 7.2 dropped gfx906 from the rocblas Tensile kernel library.
# The gfx908 kernels are architecturally close enough (both are GCN/CDNA)
# to work for the operations that still route through rocblas.
#
# Run with sudo:
#   sudo ./create-tensile-symlinks.sh
# ============================================================================
set -euo pipefail

ROCBLAS_LIB="/opt/rocm/lib/rocblas/library"

if [ ! -d "${ROCBLAS_LIB}" ]; then
    echo "ERROR: ${ROCBLAS_LIB} not found. Is ROCm installed?"
    exit 1
fi

echo "Creating gfx906 Tensile symlinks in ${ROCBLAS_LIB}..."

count=0
cd "${ROCBLAS_LIB}"
for f in *gfx908*; do
    target=$(echo "$f" | sed 's/gfx908/gfx906/g')
    if [ ! -e "$target" ]; then
        ln -sf "$f" "$target"
        ((count++))
    fi
done

echo "Created ${count} symlinks."
