# RunPod Serverless Worker for LTX-2.3 Video Generation
#
# Uses runpod/base (NOT nvidia/cuda) — includes RunPod's serverless infrastructure.
# Models auto-download on first job, cached by FlashBoot after that.

FROM runpod/base:1.0.3-cuda1281-ubuntu2204

ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install system deps
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager, used by LTX-2)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Clone LTX-2 repo and install all dependencies into the system Python
WORKDIR /app
RUN git clone --depth 1 https://github.com/Lightricks/LTX-2.git

WORKDIR /app/LTX-2
RUN uv sync --extra xformers || uv sync
RUN uv pip install runpod requests huggingface_hub

# Copy handler
COPY handler.py /app/handler.py

WORKDIR /app
CMD ["/app/LTX-2/.venv/bin/python", "-u", "/app/handler.py"]
