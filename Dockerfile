FROM runpod/base:1.0.3-cuda1281-ubuntu2204

ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app
RUN git clone --depth 1 https://github.com/Lightricks/LTX-2.git

WORKDIR /app/LTX-2
RUN uv sync --extra xformers || uv sync
RUN uv pip install runpod requests huggingface_hub hf_transfer

# Install PyTorch nightly with cu128 — includes pre-built sm_120 kernels for Blackwell GPUs
# Without this, standard PyTorch JIT-compiles CUDA kernels on first run (takes >20 min)
ENV TORCH_CUDA_ARCH_LIST="12.0"
RUN /app/LTX-2/.venv/bin/pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall --no-deps 2>/dev/null || echo "PyTorch nightly install attempted"

# Verify venv has ltx_pipelines — fail the build loudly if uv sync broke
RUN /app/LTX-2/.venv/bin/python -c "from ltx_pipelines.distilled import DistilledPipeline; print('ltx_pipelines OK')"

COPY handler.py /app/handler.py

WORKDIR /app
CMD ["/app/LTX-2/.venv/bin/python", "-u", "/app/handler.py"]
