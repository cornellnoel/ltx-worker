"""
RunPod Serverless Handler for LTX-2.3 Video Generation.

Uses DistilledPipeline (8-step inference) from Lightricks/LTX-2.
Models loaded from network volume at /runpod-volume/models/.
"""

import os
import sys
import time
import random
import tempfile

import torch
import runpod

# Ensure LTX-2 packages are importable
LTX_REPO = os.getenv("LTX_REPO_PATH", "/app/LTX-2")
sys.path.insert(0, LTX_REPO)

from ltx_core.quantization import QuantizationPolicy
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.args import ImageConditioningInput

# ── Config ──────────────────────────────────────────────────────────────────

MODEL_ROOT = os.getenv("MODEL_ROOT", "/runpod-volume/models")

# Resolution map: (resolution, aspect_ratio) → (width, height)
# All dimensions divisible by 64 for clean two-stage processing
RESOLUTION_MAP = {
    ("1080p", "16:9"): (1920, 1088),
    ("1080p", "9:16"): (1088, 1920),
    ("1440p", "16:9"): (2560, 1472),
    ("1440p", "9:16"): (1472, 2560),
    ("2160p", "16:9"): (3840, 2176),
    ("2160p", "9:16"): (2176, 3840),
}

# ── Model Loading (runs once on worker startup, stays in VRAM) ──────────────

print("[LTX] Loading DistilledPipeline...")
load_start = time.time()

pipeline = DistilledPipeline(
    distilled_checkpoint_path=os.path.join(MODEL_ROOT, "ltx-2.3-22b-distilled.safetensors"),
    gemma_root=os.path.join(MODEL_ROOT, "gemma-3-12b-it"),
    spatial_upsampler_path=os.path.join(MODEL_ROOT, "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"),
    loras=[],
    device=torch.device("cuda"),
    quantization=QuantizationPolicy.fp8_cast(),
)

print(f"[LTX] Pipeline loaded in {time.time() - load_start:.1f}s")


# ── Helpers ─────────────────────────────────────────────────────────────────

def compute_num_frames(duration: float, fps: int) -> int:
    """Compute frame count rounded to nearest valid value (8n + 1)."""
    target = int(duration * fps)
    n = round((target - 1) / 8)
    return max(8 * n + 1, 9)


def resolve_dimensions(resolution: str, aspect_ratio: str) -> tuple[int, int]:
    """Map resolution + aspect_ratio to (width, height) pixels."""
    return RESOLUTION_MAP.get((resolution, aspect_ratio), (1920, 1088))


def download_image(url: str) -> str:
    """Download image from URL to temp file, return path."""
    import requests as req

    resp = req.get(url, timeout=60)
    resp.raise_for_status()
    ext = ".png" if "png" in url.lower() else ".jpg"
    fd, path = tempfile.mkstemp(suffix=ext)
    os.write(fd, resp.content)
    os.close(fd)
    return path


def cleanup_files(*paths):
    """Silently remove temp files."""
    for p in paths:
        if p:
            try:
                os.unlink(p)
            except OSError:
                pass


# ── Handler ─────────────────────────────────────────────────────────────────

def handler(job):
    """Process a video generation request."""
    job_input = job["input"]
    job_id = job.get("id", "unknown")

    # Parse input
    prompt = job_input.get("prompt", "").strip()
    if not prompt:
        return {"error": "prompt is required"}

    duration = float(job_input.get("duration", 6))
    resolution = job_input.get("resolution", "1080p")
    fps = int(job_input.get("fps", 25))
    aspect_ratio = job_input.get("aspect_ratio", "16:9")
    generate_audio = job_input.get("generate_audio", True)
    seed = job_input.get("seed") or random.randint(1, 999999)
    enhance_prompt = job_input.get("enhance_prompt", False)
    image_url = job_input.get("image_url")
    end_image_url = job_input.get("end_image_url")

    width, height = resolve_dimensions(resolution, aspect_ratio)
    num_frames = compute_num_frames(duration, fps)

    print(f"[LTX] Job {job_id}: {width}x{height}, {num_frames}f @ {fps}fps, seed={seed}")

    # Image conditioning (i2v mode)
    images = []
    temp_files = []

    if image_url:
        try:
            path = download_image(image_url)
            temp_files.append(path)
            images.append(ImageConditioningInput(path=path, frame_idx=0, strength=1.0))
        except Exception as e:
            cleanup_files(*temp_files)
            return {"error": f"Failed to download start image: {e}"}

    if end_image_url:
        try:
            path = download_image(end_image_url)
            temp_files.append(path)
            images.append(ImageConditioningInput(path=path, frame_idx=num_frames - 1, strength=1.0))
        except Exception as e:
            cleanup_files(*temp_files)
            return {"error": f"Failed to download end image: {e}"}

    # Generate
    gen_start = time.time()
    output_path = f"/tmp/ltx_{job_id}.mp4"

    try:
        tiling_config = TilingConfig.default()
        video_chunks = get_video_chunks_number(num_frames, tiling_config)

        runpod.serverless.progress_update(job, "Generating video...")

        video, audio = pipeline(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=float(fps),
            images=images,
            tiling_config=tiling_config,
            enhance_prompt=enhance_prompt,
        )

        runpod.serverless.progress_update(job, "Encoding video...")

        encode_video(
            video=video,
            fps=fps,
            audio=audio if generate_audio else None,
            output_path=output_path,
            video_chunks_number=video_chunks,
        )

        gen_time = time.time() - gen_start
        print(f"[LTX] Generated in {gen_time:.1f}s")

    except Exception as e:
        cleanup_files(output_path, *temp_files)
        return {"error": f"Generation failed: {str(e)}"}

    # Upload to S3 (RunPod's built-in bucket, URLs valid for 7 days)
    video_url = None
    try:
        from runpod.serverless.utils import rp_upload

        video_url = rp_upload.upload_image(job_id, output_path)
    except Exception as e:
        print(f"[LTX] S3 upload failed: {e}")

    # Cleanup
    cleanup_files(output_path, *temp_files)

    if not video_url:
        return {"error": "Failed to upload generated video"}

    return {
        "video_url": video_url,
        "duration": duration,
        "width": width,
        "height": height,
        "fps": fps,
        "num_frames": num_frames,
        "seed": seed,
        "generation_time_seconds": round(gen_time, 1),
    }


runpod.serverless.start({"handler": handler})
