# ltx-worker

RunPod Serverless worker for [LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) video generation.

Uses the DistilledPipeline (8-step inference) with FP8 quantization on 80GB GPUs.

## How it works

1. Worker starts instantly — no model loading at startup
2. First job triggers model download (~43GB distilled model + ~24GB Gemma 3 text encoder)
3. RunPod FlashBoot caches the loaded state for fast subsequent cold starts (~2-5s)

## API

### Submit a job

```bash
curl -X POST "https://api.runpod.ai/v2/{endpoint_id}/run" \
  -H "Authorization: Bearer {your_runpod_api_key}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "A golden retriever on a beach at sunset",
      "duration": 6,
      "resolution": "1080p",
      "fps": 25,
      "aspect_ratio": "16:9"
    }
  }'
```

### Poll for result

```bash
curl "https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}" \
  -H "Authorization: Bearer {your_runpod_api_key}"
```

### Input

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | *required* | Video description |
| `duration` | float | 6 | Duration in seconds |
| `resolution` | string | "1080p" | "1080p", "1440p", or "2160p" |
| `fps` | int | 25 | Frames per second |
| `aspect_ratio` | string | "16:9" | "16:9" or "9:16" |
| `generate_audio` | bool | true | Generate audio track |
| `seed` | int | random | Reproducibility seed |
| `enhance_prompt` | bool | false | LLM prompt enhancement |
| `image_url` | string | null | Start frame URL (image-to-video) |
| `end_image_url` | string | null | End frame URL |

### Output

```json
{
  "video_url": "https://...",
  "width": 1920,
  "height": 1088,
  "fps": 25,
  "num_frames": 151,
  "seed": 42,
  "generation_time_seconds": 45.2
}
```

## Deploy

### Prerequisites

- RunPod account with 80GB GPU tier enabled (A100/H100)
- GitHub repo with Actions enabled (builds Docker image automatically)

### Setup

1. Push to `main` — GitHub Actions builds and pushes the Docker image to GHCR
2. Create a Serverless endpoint in RunPod console:
   - Container image: `ghcr.io/cornellnoel/ltx-worker:latest`
   - GPU: 80GB (48GB will OOM)
   - Execution timeout: 600s+
   - Container disk: 150GB
3. First job will take ~15-20 min (model download). Enable FlashBoot after first success.

### Updating

After pushing code changes:
1. Wait for GitHub Actions build to complete (~25-35 min)
2. Cycle workers: set Max Workers to 0 → Save → wait 30s → set to 1 → Save
3. New worker picks up the updated image

## Architecture

```
Job request → RunPod Serverless → handler.py
                                    ├── Lazy-load DistilledPipeline (first job only)
                                    ├── Download models from HuggingFace (first job only)
                                    ├── Generate video (8-step distilled inference)
                                    ├── Encode MP4 with optional audio
                                    └── Upload to RunPod S3 → return video_url
```

## Known issues

- LTX-2 has a circular import: `DistilledPipeline` must be imported before `QuantizationPolicy`
- 48GB GPUs OOM loading the model — 80GB minimum required
- First cold start takes 15-20 min (model download); FlashBoot eliminates this after first run
