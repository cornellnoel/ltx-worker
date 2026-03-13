"""
RunPod Serverless Handler for LTX-2.3 Video Generation.

Supports two modes:
  - "fast" (default): DistilledPipeline, 8-step inference, ~30s generation
  - "lora": Dev model with LoRA weights, higher quality, ~2min generation

Models auto-downloaded on first job, then cached by FlashBoot.

Key design: runpod.serverless.start() is called IMMEDIATELY at startup
(no model loading). Models download and load lazily on the first job.
This avoids RunPod's worker initialization timeout.
"""

import os
import sys
import time
import random
import tempfile

import runpod

# ── Config ──────────────────────────────────────────────────────────────────

MODEL_ROOT = os.getenv("MODEL_ROOT", "/runpod-volume/models")
LTX_REPO = os.getenv("LTX_REPO_PATH", "/app/LTX-2")

RESOLUTION_MAP = {
    ("480p", "16:9"): (848, 480),
    ("480p", "9:16"): (480, 848),
    ("480p", "1:1"): (512, 512),
    ("720p", "16:9"): (1280, 720),
    ("720p", "9:16"): (720, 1280),
    ("720p", "1:1"): (768, 768),
    ("1080p", "16:9"): (1920, 1088),
    ("1080p", "9:16"): (1088, 1920),
    ("1080p", "1:1"): (1088, 1088),
    ("1440p", "16:9"): (2560, 1472),
    ("1440p", "9:16"): (1472, 2560),
    ("2160p", "16:9"): (3840, 2176),
    ("2160p", "9:16"): (2176, 3840),
}

MODEL_FILES = {
    "ltx-2.3-22b-distilled.safetensors": "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled.safetensors",
    "ltx-2.3-spatial-upscaler-x2-1.0.safetensors": "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
}
GEMMA_REPO = "unsloth/gemma-3-12b-it"

# Global state
pipeline = None
current_mode = None


# ── Model Download + Load ───────────────────────────────────────────────────

def download_file(url, dest):
    """Download a large file with progress logging."""
    import requests as req

    print(f"[LTX]   Downloading to {dest}...")
    resp = req.get(url, stream=True, timeout=1800)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=64 * 1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded * 100 // total
                print(f"[LTX]   {downloaded // (1024*1024)}MB / {total // (1024*1024)}MB ({pct}%)")


def ensure_models():
    """Download models if not present."""
    os.makedirs(MODEL_ROOT, exist_ok=True)

    for filename, url in MODEL_FILES.items():
        path = os.path.join(MODEL_ROOT, filename)
        if os.path.exists(path):
            print(f"[LTX] Found {filename}")
            continue
        print(f"[LTX] Downloading {filename}...")
        download_file(url, path)
        print(f"[LTX] Downloaded {filename}")

    gemma_dir = os.path.join(MODEL_ROOT, "gemma-3-12b-it")
    if os.path.isdir(gemma_dir) and os.listdir(gemma_dir):
        print("[LTX] Found Gemma 3 text encoder")
    else:
        print("[LTX] Downloading Gemma 3 12B text encoder...")
        from huggingface_hub import snapshot_download

        snapshot_download(GEMMA_REPO, local_dir=gemma_dir)
        print("[LTX] Downloaded Gemma 3 text encoder")


def load_pipeline(mode="fast", loras=None):
    """Load or swap the pipeline based on mode."""
    global pipeline, current_mode
    import torch

    if pipeline is not None and current_mode == mode:
        return

    # Unload existing pipeline if switching modes
    if pipeline is not None:
        print(f"[LTX] Unloading {current_mode} pipeline to switch to {mode}...")
        del pipeline
        pipeline = None
        torch.cuda.empty_cache()

    sys.path.insert(0, LTX_REPO)
    # Import DistilledPipeline FIRST — it resolves a circular import
    # between ltx_core.quantization and ltx_core.loader
    from ltx_pipelines.distilled import DistilledPipeline
    from ltx_core.quantization import QuantizationPolicy

    ensure_models()

    if mode == "fast":
        print("[LTX] Loading DistilledPipeline (fast mode)...")
        load_start = time.time()

        pipeline = DistilledPipeline(
            distilled_checkpoint_path=os.path.join(MODEL_ROOT, "ltx-2.3-22b-distilled.safetensors"),
            gemma_root=os.path.join(MODEL_ROOT, "gemma-3-12b-it"),
            spatial_upsampler_path=os.path.join(MODEL_ROOT, "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"),
            loras=[],
            device=torch.device("cuda"),
            quantization=QuantizationPolicy.fp8_cast(),
        )

        print(f"[LTX] Fast pipeline loaded in {time.time() - load_start:.1f}s")
        current_mode = "fast"

    elif mode == "lora":
        # TODO: Dev model + LoRA support
        # Will use the dev checkpoint with user-specified LoRA weights
        print("[LTX] LoRA mode not yet implemented, falling back to fast mode")
        load_pipeline("fast")
        return

    else:
        raise ValueError(f"Unknown mode: {mode}")


# ── Helpers ─────────────────────────────────────────────────────────────────

def compute_num_frames(duration, fps):
    target = int(duration * fps)
    n = round((target - 1) / 8)
    return max(8 * n + 1, 9)


def resolve_dimensions(resolution, aspect_ratio):
    dims = RESOLUTION_MAP.get((resolution, aspect_ratio))
    if dims is None:
        print(f"[LTX] WARNING: No resolution entry for ({resolution}, {aspect_ratio}), falling back to 720p 16:9", flush=True)
        dims = (1280, 720)
    return dims


def download_image(url):
    import requests as req

    resp = req.get(url, timeout=60)
    resp.raise_for_status()
    ext = ".png" if "png" in url.lower() else ".jpg"
    fd, path = tempfile.mkstemp(suffix=ext)
    os.write(fd, resp.content)
    os.close(fd)
    return path


def cleanup_files(*paths):
    for p in paths:
        if p:
            try:
                os.unlink(p)
            except OSError:
                pass


# ── Handler ─────────────────────────────────────────────────────────────────

def handler(job):
    """Process a video generation request. Lazy-loads models on first call."""
    try:
        return _handler_inner(job)
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print(f"[LTX] FATAL handler error: {e}", flush=True)
        print(f"[LTX] Traceback:\n{err}", flush=True)
        return {"error": f"Handler crashed: {str(e)}", "traceback": err}


def _handler_inner(job):
    job_input = job["input"]
    job_id = job.get("id", "unknown")

    prompt = job_input.get("prompt", "").strip()
    if not prompt:
        return {"error": "prompt is required"}

    mode = job_input.get("mode", "fast")

    # Lazy load pipeline on first job (or swap if mode changed)
    load_pipeline(mode)

    # Import LTX-2 modules (available after load_pipeline sets up sys.path)
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_pipelines.utils.media_io import encode_video
    from ltx_pipelines.utils.args import ImageConditioningInput

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

    print(f"[LTX] Job {job_id}: {width}x{height}, {num_frames}f @ {fps}fps, seed={seed}, mode={mode}")

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

    # Upload to S3
    video_url = None
    try:
        from runpod.serverless.utils import rp_upload

        video_url = rp_upload.upload_image(job_id, output_path)
    except Exception as e:
        print(f"[LTX] S3 upload failed: {e}")

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


# ── Start handler immediately (no model loading at startup) ─────────────────

print("[LTX] Handler starting (models load on first job)...", flush=True)

# Log GPU/CUDA info at startup for diagnostics
try:
    import torch
    print(f"[LTX] PyTorch {torch.__version__}", flush=True)
    print(f"[LTX] CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"[LTX] GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"[LTX] VRAM: {torch.cuda.get_device_properties(0).total_memory // (1024**3)}GB", flush=True)
except Exception as e:
    print(f"[LTX] GPU info failed: {e}", flush=True)

runpod.serverless.start({"handler": handler})
