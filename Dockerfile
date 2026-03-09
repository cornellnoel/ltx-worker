# RunPod Serverless Worker for LTX-2.3 Video Generation
#
# Models are NOT in this image — they load from a network volume at /runpod-volume/models/
# This keeps the image small (~8-10GB) and rebuilds fast.
#
# Build:   docker build -t ltx-worker .
# Push:    docker tag ltx-worker <registry>/ltx-worker:latest && docker push <registry>/ltx-worker:latest
# Deploy:  Create Serverless endpoint on RunPod, attach network volume, set Docker image

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Python 3.12 from deadsnakes PPA (LTX-2 requires >= 3.12)
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        python3.12 python3.12-dev python3.12-venv \
        git curl wget ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Install uv (fast Python package manager, used by LTX-2)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Clone LTX-2 repo and install all dependencies
# Pin to a specific commit for reproducibility (update as needed)
WORKDIR /app
RUN git clone --depth 1 https://github.com/Lightricks/LTX-2.git

WORKDIR /app/LTX-2
RUN uv sync --extra xformers || uv sync
RUN uv pip install runpod requests

# Copy handler
COPY handler.py /app/handler.py

WORKDIR /app
CMD ["/app/LTX-2/.venv/bin/python", "-u", "/app/handler.py"]
