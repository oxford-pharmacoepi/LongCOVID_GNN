# ============================================================
# Dockerfile for LongCOVID GNN / SEAL Training Pipeline
# Uses NVIDIA CUDA 12.1 runtime with Python 3.12 via uv
# Target: Google Cloud Platform (GCE / Vertex AI)
# ============================================================

FROM --platform=linux/amd64 nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# ---- System dependencies (minimal — uv handles Python) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---- Install uv (fast Python package manager) ----
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# ---- Install Python 3.12 via uv (no PPA needed) ----
RUN uv python install 3.12

# ---- Working directory ----
WORKDIR /app

# ---- Copy project metadata first (for layer caching) ----
COPY pyproject.toml uv.lock* README.md* ./

# ---- Install heavy dependencies via uv (cached layer) ----
# Install PyTorch with CUDA 12.1 support
RUN uv venv --python 3.12 \
    && uv pip install --no-cache \
    torch>=2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121 \
    && uv pip install --no-cache \
    torch-geometric>=2.7.0 \
    torch-scatter \
    torch-sparse \
    -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

# ---- Copy project source ----
COPY src/ src/
COPY scripts/ scripts/

# ---- Install project in editable mode (needs src/) ----
RUN uv pip install --no-cache -e .

# ---- Create output directories ----
RUN mkdir -p results raw_data processed_data

# ---- Volume mounts for data I/O ----
# Mount your local data: -v /path/to/data:/app/data
# Mount results out:     -v /path/to/output:/app/results
VOLUME ["/app/results", "/app/raw_data", "/app/processed_data"]

# ---- Entrypoint: flexible — run any script ----
# Examples:
#   SEAL LOO:    docker run <image> seal-loo --target-disease EFO_0003854 --epochs 50
#   GNN train:   docker run <image> gnn-train
#   GNN LOO:     docker run <image> gnn-loo
#   Custom:      docker run <image> python scripts/my_script.py --args
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["--help"]
