# FrameSeg

FrameSeg is a self-hosted, AI-assisted rotoscoping studio powered by Meta's SAM2.1
video segmentation models. Select a subject on one frame, refine the generated
whole-object mask, and track it in both directions through a chosen section of a
video. The result is an alpha-channel video ready for compositing or editing.

Professional editing applications often place their fastest AI masking and
rotoscoping workflows behind paid products or higher-tier editions. FrameSeg explores
a more accessible path: run SAM2.1 on your own NVIDIA GPU, keep the footage under your
control, and use the result in the editor of your choice.

> FrameSeg is an independent project and is not affiliated with Meta, Blackmagic
> Design, or DaVinci Resolve. Review the terms of the SAM2.1 model you choose before
> using it in production.

## What it does

- Accepts H.264 and H.265 video in MP4 or MOV containers.
- Lets the user choose any consecutive 30–1,200-frame range on the full timeline.
- Creates a whole-object mask from a point or bounding-box prompt.
- Supports positive and negative clicks to add to or subtract from the mask.
- Provides scroll-wheel zoom and <kbd>Space</kbd> + drag panning for fine selection.
- Tracks from the seed frame both backward and forward through the selected range.
- Offers the SAM2.1 Hiera Tiny, Small, Base+, and Large model family before rendering.
- Streams decoding, inference, and alpha encoding through bounded queues instead of
  retaining an entire 1,200-frame clip in memory.
- Exports a transparent ProRes 4444 MOV by default, with QTRLE MOV and VP9-alpha WebM
  available as configurable alternatives.
- Clears the prior video's upload, mask state, cached frames, and output when starting
  another video, while keeping the loaded default model available.

## How the pipeline works

```text
Browser
  │ upload + range + mask corrections
  ▼
Next.js studio ───────► FastAPI service
                           │
              PyAV decode ─┤─ FFmpeg fallback
                           ▼
                 SAM2.1 selection + tracking
                           │
                  bounded frame/mask queues
                           ▼
                   FFmpeg alpha-video export
```

PyAV handles normal decoding, while FFmpeg provides a fallback for troublesome
H.264/H.265 MP4 and MOV inputs. During rendering, decoded frames are delivered to
SAM2.1 incrementally. Completed masks are encoded immediately, and frame references
are released as the pipeline advances. Debug RGB, alpha, and RGBA PNG sequences are
disabled by default.

## Requirements

For the Docker setup:

- Linux with an NVIDIA GPU and a current NVIDIA driver
- Docker Engine with the Compose plugin
- NVIDIA Container Toolkit configured for Docker
- Enough storage for the model cache, source footage, and alpha output

For native development, also install:

- Python 3.14 and [`uv`](https://docs.astral.sh/uv/)
- FFmpeg and FFprobe
- Node.js 24 and npm

CPU execution is supported by the backend code but is not practical for production
video tracking. The locked backend environment currently uses PyTorch 2.14 with CUDA
13 wheels.

## Quick start with published images

Clone the repository and create the local Compose configuration:

```bash
git clone https://github.com/Ioncrade/Frameseg.git
cd Frameseg
cp .env.docker.example .env
```

The defaults expose the frontend at `http://localhost:3000` and the API at
`http://localhost:8000`. Pull and start the published Linux AMD64 images:

```bash
docker compose pull
docker compose up -d
docker compose ps
```

The first backend start downloads the default SAM2.1 checkpoint into the persistent
`frameseg-model-cache` volume. Follow startup progress with:

```bash
docker compose logs -f backend
```

Stop the application without deleting uploads or the model cache:

```bash
docker compose down
```

To remove the persistent data and downloaded model cache as well:

```bash
docker compose down --volumes
```

Published images:

- `ioncrade/openframe-backend:latest`
- `ioncrade/openframe-frontend:latest`

## Build the containers locally

Set `NEXT_PUBLIC_API_URL` in `.env` to the browser-accessible backend address before
building. Next.js embeds this public value into the frontend bundle at build time.

```bash
docker compose build
docker compose up -d
```

For deployment on another host, use public values such as:

```dotenv
NEXT_PUBLIC_API_URL=https://api.example.com
FRONTEND_ORIGINS=https://app.example.com
```

Then rebuild the frontend image. `FRONTEND_ORIGINS` accepts the origins allowed to
call the backend.

## Native development

Start the backend:

```bash
cd backend
uv sync
uv run python main.py
```

In another terminal, start the frontend:

```bash
cd frontend
cp .env.example .env.local
npm ci
npm run dev
```

Open `http://localhost:3000`. FastAPI documentation is available at
`http://localhost:8000/docs`, and backend health is reported at
`http://localhost:8000/api/health`.

## Using FrameSeg

1. Upload an H.264/H.265 MP4 or MOV video.
2. Move and resize the range directly on the timeline. The selected window must be
   between 30 and 1,200 frames.
3. Choose a seed frame inside the range and click the subject. Scroll to zoom; hold
   <kbd>Space</kbd> and drag to pan around a zoomed frame.
4. Refine the full-object overlay with **Add** and **Subtract** clicks.
5. Pick a SAM2.1 model and start the render.
6. Download the transparent output. The accepted mask remains available so the same
   clip can be rendered again with another model.
7. Use **Add another video** to wipe the current video's working state and begin a new
   project without reloading the default model.

## Model choices

| Option | Hugging Face model | Profile |
| --- | --- | --- |
| Tiny | `facebook/sam2.1-hiera-tiny` | Fastest |
| Small | `facebook/sam2.1-hiera-small` | Fast |
| Base+ | `facebook/sam2.1-hiera-base-plus` | Balanced default |
| Large | `facebook/sam2.1-hiera-large` | Highest quality, slowest |

The first use of a variant may download its checkpoint. Non-default tracking models
are unloaded after their job so switching variants does not continuously accumulate
VRAM. Render jobs are admitted one at a time per GPU.

## Output formats

Set `FRAMESEG_OUTPUT_CODEC` in `.env` before starting the backend:

| Value | Output | Use case |
| --- | --- | --- |
| `prores_4444` | Alpha MOV | Default; broad NLE/VFX workflow compatibility |
| `qtrle` | Lossless alpha MOV | Fast encoding and exact alpha, potentially larger |
| `vp9_alpha` | Compressed alpha WebM | Smaller delivery file with narrower editor support |

FrameSeg clears invisible RGB data before encoding, which avoids spending most of the
output bitrate on pixels hidden by a zero alpha value. The output contains the selected
video range and does not currently preserve the source audio track.

## Configuration

The main Compose options live in `.env.docker.example`:

| Variable | Default | Purpose |
| --- | --- | --- |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Browser-visible backend URL, embedded at frontend build time |
| `FRONTEND_ORIGINS` | `http://localhost:3000` | Comma-separated CORS origins |
| `FRONTEND_PORT` | `3000` | Published frontend port |
| `BACKEND_PORT` | `8000` | Published backend port |
| `HF_TOKEN` | empty | Optional Hugging Face access token |
| `FRAMESEG_PRELOAD_MODEL` | `true` | Load the default model during API startup |
| `FRAMESEG_OFFLOAD_STATE_TO_CPU` | `false` | Reduce VRAM use at the cost of transfer overhead |
| `FRAMESEG_OUTPUT_CODEC` | `prores_4444` | Alpha-video encoder profile |

Additional tuning options, API endpoints, and output details are documented in
[`backend/README.md`](backend/README.md). The performance investigation and design
decisions are recorded in [`optimization.md`](optimization.md).

## Repository layout

```text
backend/             FastAPI API, media loader, SAM2 service, and alpha encoder
frontend/            Next.js selection and timeline studio
docker-compose.yml   GPU-enabled local/deployment stack
optimization.md      Throughput, memory, and output-size engineering notes
```

The former `testscripts/` prototype workspace and all generated media are deliberately
excluded from version control.

## Development checks

```bash
cd frontend
npm run lint
npm run build

cd ../backend
uv run python -m compileall -q .

cd ..
docker compose config
docker build --check backend
docker build --check frontend
```

Model inference requires a compatible NVIDIA GPU and downloaded weights, so it is best
validated with a short real clip after the static checks pass.

## Current limitations

- Projects and job records are held in backend memory and do not survive a restart.
- The interactive range is capped at 1,200 frames per render.
- One render job runs at a time on a GPU.
- The Docker images currently target Linux AMD64 with NVIDIA acceleration.
- Authentication, multi-user isolation, persistent job metadata, and automatic
  artifact retention policies are not implemented yet.

Contributions that improve tracking quality, resource scheduling, export
compatibility, or production durability are welcome.

## License

FrameSeg is licensed under the Apache License 2.0. See [`LICENSE`](LICENSE)
for the full license terms.
