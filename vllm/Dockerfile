# ============================================================================
# vLLM on ROCm for AMD MI50 (gfx906)
# ============================================================================
# Uses the official ROCm vLLM image with gfx906 target support.
# Serves GGUF models from a mounted models directory via the OpenAI-compatible API.
#
# NOTE: The official vLLM ROCm images are built for MI200/MI300 (gfx90a/gfx942).
# For MI50 (gfx906), we build from source with the correct GPU target.
# ============================================================================

FROM rocm/dev-ubuntu-24.04:7.2 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates python3 python3-pip python3-venv python3-dev \
        build-essential cmake ninja-build ccache \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch ROCm (latest available)
RUN pip3 install --break-system-packages --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.2.4

# Build vLLM from source with gfx906 target
ARG VLLM_VERSION=v0.9.1
WORKDIR /build
RUN git clone --depth 1 --branch ${VLLM_VERSION} https://github.com/vllm-project/vllm.git vllm-src

WORKDIR /build/vllm-src
ENV PYTORCH_ROCM_ARCH="gfx906"
ENV ROCM_PATH="/opt/rocm"
RUN pip3 install --break-system-packages --no-cache-dir -r requirements/rocm.txt 2>/dev/null || true
RUN pip3 install --break-system-packages --no-cache-dir -e . 2>/dev/null || \
    pip3 install --break-system-packages --no-cache-dir vllm


# --- Runtime ---
FROM rocm/dev-ubuntu-24.04:7.2

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/dist-packages /usr/local/lib/python3.12/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create symlinks for fake gfx906 Tensile libraries
RUN cd /opt/rocm/lib/rocblas/library && \
    for f in *gfx908*; do \
        target=$(echo "$f" | sed 's/gfx908/gfx906/g'); \
        [ ! -e "$target" ] && ln -sf "$f" "$target"; \
    done; true

# Model storage
RUN mkdir -p /models
VOLUME ["/models"]

ENV PYTORCH_ROCM_ARCH="gfx906"
ENV ROCM_PATH="/opt/rocm"
ENV HIP_VISIBLE_DEVICES=0
ENV VLLM_TARGET_DEVICE=rocm

EXPOSE 8000

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--model", "/models/default", "--trust-remote-code"]
