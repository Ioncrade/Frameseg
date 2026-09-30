# FrameSeg backend

FastAPI service for uploading short H.264/H.265 clips, selecting and refining an
object with SAM2, tracking it through the video, and exporting a transparent
alpha-channel video. The default is a quality-first ProRes 4444 MOV at the source
resolution and frame rate. QTRLE MOV and compressed VP9-alpha WebM remain optional.

## Run locally

Requirements: Python 3.14, `uv`, FFmpeg/FFprobe, and an NVIDIA driver compatible
with the CUDA 13 PyTorch wheels.

```bash
uv sync
uv run python main.py
```

The API starts at `http://localhost:8000`. The interactive API schema is at
`http://localhost:8000/docs` and health is available at `/api/health`.

The service preloads `facebook/sam2.1-hiera-base-plus` at startup. The first launch may
download the model if it is not already cached.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `FRAMESEG_HOST` | `0.0.0.0` | Bind address |
| `FRAMESEG_PORT` | `8000` | API port |
| `FRAMESEG_RELOAD` | `false` | Enable Uvicorn reload mode |
| `FRAMESEG_DATA_ROOT` | `backend/data` | Uploaded footage, masks, and outputs |
| `FRAMESEG_MAX_UPLOAD_BYTES` | `2147483648` | Maximum upload size |
| `FRONTEND_ORIGINS` | `http://localhost:3000` | Comma-separated CORS origins |
| `FRAMESEG_MODEL_ID` | `facebook/sam2.1-hiera-base-plus` | SAM2 video checkpoint |
| `FRAMESEG_PRELOAD_MODEL` | `true` | Load the checkpoint during API startup |
| `FRAMESEG_OFFLOAD_STATE_TO_CPU` | `false` | Save VRAM at the cost of per-frame transfers |
| `FRAMESEG_DECODE_PREFETCH` | `3` | Bounded decoded-frame queue depth |
| `FRAMESEG_ENCODE_QUEUE_DEPTH` | `3` | Bounded alpha-encoder queue depth |
| `FRAMESEG_OUTPUT_CODEC` | `prores_4444` | `prores_4444` or `qtrle` MOV, or `vp9_alpha` WebM |
| `FRAMESEG_ENCODER_THREADS` | `0` | Encoder threads; `0` lets FFmpeg choose |
| `FRAMESEG_VP9_CRF` | `18` | VP9 quality; lower is higher quality and larger (0–63) |
| `FRAMESEG_VP9_CPU_USED` | `5` | VP9 speed/efficiency tradeoff (0–8); higher is faster |
| `FRAMESEG_REVERSE_DECODE_CHUNK` | `16` | Frames buffered while decoding backward (1–64) |
| `FRAMESEG_KEEP_FRAME_ARTIFACTS` | `false` | Keep RGB/alpha/RGBA debug PNGs |

Users select a consecutive range of 30–1,200 frames anywhere in the source video.
Only that interval is decoded for SAM2 tracking. Decoding uses PyAV first
and falls back to FFmpeg for supported H.264/H.265 MP4 and MOV inputs.
Frames and masks move through bounded queues and are released as they are encoded;
the normal path does not retain a 1,200-frame PNG sequence.

## API flow

1. `POST /api/videos` with a multipart `file`.
2. Use `GET /api/videos/{id}/frame.png?frame_index=N` to preview the timeline and
   choose a start frame plus a 30–1,200-frame window.
3. `POST /api/selections` with the source-space point or bounding box,
   `start_frame`, `frame_count`, and a `seed_frame` inside that range.
4. `PATCH /api/selections/{id}` with positive or negative correction points.
5. `POST /api/selections/{id}/render` to track backward and forward from the seed.
6. Poll `GET /api/jobs/{id}`, then download `/api/jobs/{id}/output`.

The render request accepts `model_variant` as `tiny`, `small`, `base_plus`, or
`large`. `GET /api/models` returns the supported SAM2.1 family and the default. Render
jobs are admitted one at a time per GPU. Non-default tracking checkpoints are unloaded
when their job finishes so switching models does not accumulate VRAM. The first render
with a variant may download its checkpoint from Hugging Face; later renders use the
local model cache. The accepted mask is retained so the same selection can be rendered
again with another model variant.

Runtime records are currently held in memory. Uploaded files and completed
artifacts remain on disk, but API job IDs do not survive a backend restart.
