# ============================================================================
# Ollama on ROCm 7.2 for AMD MI50 (gfx906)
# ============================================================================
# This Dockerfile builds a patched libggml-hip.so that replaces the one shipped
# with Ollama, enabling full GPU acceleration on MI50 under ROCm 7.2.
#
# The official Ollama binary and rocBLAS libraries are used as-is; only the
# GGML HIP backend is rebuilt with gfx906-specific patches.
# ============================================================================

# --- Stage 1: Build patched libggml-hip.so ---
FROM rocm/dev-ubuntu-24.04:7.2 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build ccache golang-go curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ARG OLLAMA_VERSION=v0.17.7
WORKDIR /build

# Clone Ollama source at the exact version
RUN git clone --depth 1 --branch ${OLLAMA_VERSION} https://github.com/ollama/ollama.git ollama-src

# Apply gfx906 patches
COPY patches/gfx906-rocm72-all.diff /build/patches/
RUN cd ollama-src && git apply /build/patches/gfx906-rocm72-all.diff

# Build only the patched ggml-hip library
RUN cd ollama-src && cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DAMDGPU_TARGETS="gfx906" \
        -DGGML_CUDA_FORCE_MMQ=ON \
        -DGGML_HIP_NO_MMQ_MFMA=ON \
        -DCMAKE_PREFIX_PATH="/opt/rocm" \
    && cmake --build build --target ggml-hip -j$(nproc)


# --- Stage 2: Runtime image ---
FROM rocm/dev-ubuntu-24.04:7.2

LABEL maintainer="noshitcoding"
LABEL description="Ollama with patched gfx906 (MI50) support on ROCm 7.2"

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama binary
ARG OLLAMA_VERSION=v0.17.7
RUN curl -fsSL https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/ollama-linux-amd64.tgz \
    | tar xz -C /usr \
    && chmod +x /usr/bin/ollama

# Create symlinks for fake gfx906 Tensile libraries (gfx906 -> gfx908)
# ROCm 7.2 dropped gfx906 from rocblas Tensile kernels, but gfx908 kernels
# are close enough for the operations that still go through rocblas.
RUN cd /opt/rocm/lib/rocblas/library && \
    for f in *gfx908*; do \
        target=$(echo "$f" | sed 's/gfx908/gfx906/g'); \
        [ ! -e "$target" ] && ln -sf "$f" "$target"; \
    done; true

# Copy the patched libggml-hip.so over the Ollama-shipped one
COPY --from=builder /build/ollama-src/build/lib/ollama/libggml-hip.so /usr/lib/ollama/rocm/libggml-hip.so

# Create ollama user and model directory
RUN useradd -r -s /bin/false -m -d /usr/share/ollama ollama \
    && mkdir -p /usr/share/ollama/.ollama/models

# Runtime environment
ENV OLLAMA_HOST=0.0.0.0:11434
ENV OLLAMA_LLM_LIBRARY=rocm
ENV OLLAMA_FLASH_ATTENTION=1
ENV OLLAMA_MAX_LOADED_MODELS=1
ENV OLLAMA_KEEP_ALIVE=60s
ENV GPU_MAX_HW_QUEUES=2
ENV GGML_CUDA_INIT=1
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/usr/lib/ollama/rocm

EXPOSE 11434

VOLUME ["/usr/share/ollama/.ollama/models"]

USER ollama
ENTRYPOINT ["/usr/bin/ollama"]
CMD ["serve"]
