#!/bin/bash
# Download LTX-2.3 models to RunPod network volume.
#
# Run this on a temporary RunPod pod with the network volume mounted at /runpod-volume.
# Total download: ~70GB (distilled model + spatial upscaler + Gemma 3 text encoder)
#
# Usage:
#   1. Create a network volume on RunPod (100GB minimum, 150GB recommended)
#   2. Start a cheap pod (any GPU) with that volume mounted
#   3. SSH in and run: bash /workspace/download-models.sh
#   4. Stop the pod (volume persists)
#   5. Create Serverless endpoint pointing to the same volume

set -euo pipefail

MODEL_DIR="/runpod-volume/models"
mkdir -p "$MODEL_DIR"

echo "=== Downloading LTX-2.3 models to $MODEL_DIR ==="
echo "This will download ~70GB. Estimated time: 10-20 minutes on RunPod."
echo ""

# ── Distilled model checkpoint (~43GB) ──
FILE="$MODEL_DIR/ltx-2.3-22b-distilled.safetensors"
if [ -f "$FILE" ]; then
    echo "[SKIP] Distilled model already exists"
else
    echo "[1/3] Downloading distilled model (ltx-2.3-22b-distilled.safetensors, ~43GB)..."
    wget -q --show-progress -O "$FILE" \
        "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled.safetensors"
    echo "[1/3] Done."
fi

# ── Spatial upscaler (~few GB) ──
FILE="$MODEL_DIR/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
if [ -f "$FILE" ]; then
    echo "[SKIP] Spatial upscaler already exists"
else
    echo "[2/3] Downloading spatial upscaler (ltx-2.3-spatial-upscaler-x2-1.0.safetensors)..."
    wget -q --show-progress -O "$FILE" \
        "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
    echo "[2/3] Done."
fi

# ── Gemma 3 12B text encoder ──
# The Python pipeline expects the full HuggingFace model directory
GEMMA_DIR="$MODEL_DIR/gemma-3-12b-it"
if [ -d "$GEMMA_DIR" ] && [ "$(ls -A $GEMMA_DIR 2>/dev/null)" ]; then
    echo "[SKIP] Gemma 3 text encoder already exists"
else
    echo "[3/3] Downloading Gemma 3 12B text encoder..."

    # Check if huggingface-cli is available
    if ! command -v huggingface-cli &> /dev/null; then
        echo "  Installing huggingface_hub..."
        pip install -q huggingface_hub
    fi

    # Download with huggingface-cli (handles sharded model files)
    huggingface-cli download google/gemma-3-12b-it \
        --local-dir "$GEMMA_DIR" \
        --local-dir-use-symlinks False

    echo "[3/3] Done."
fi

echo ""
echo "=== All models downloaded ==="
echo ""
ls -lh "$MODEL_DIR/"
echo ""
du -sh "$MODEL_DIR"
echo ""
echo "Network volume is ready. You can now:"
echo "  1. Stop this pod"
echo "  2. Create a Serverless endpoint with this network volume"
echo "  3. Set the Docker image to your ltx-worker image"
