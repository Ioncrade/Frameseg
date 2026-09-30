from __future__ import annotations

import os
import shutil
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from PIL import Image
from pydantic import BaseModel, Field, model_validator
from starlette.concurrency import run_in_threadpool

from media_loader import (
    MAX_VIDEO_FRAMES,
    MIN_VIDEO_FRAMES,
    VideoMetadata,
    iter_video_frames,
    probe_video,
)
from model_service import ObjectPrompt, Sam2ModelService
from rotoscope_service import RotoscopeService


BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(os.getenv("FRAMESEG_DATA_ROOT", BASE_DIR / "data")).resolve()
UPLOAD_ROOT = DATA_ROOT / "uploads"
PREVIEW_ROOT = DATA_ROOT / "previews"
SELECTION_ROOT = DATA_ROOT / "selections"
OUTPUT_ROOT = DATA_ROOT / "outputs"
MAX_UPLOAD_BYTES = int(os.getenv("FRAMESEG_MAX_UPLOAD_BYTES", str(2 * 1024**3)))
UPLOAD_CHUNK_BYTES = 8 * 1024**2
Sam2ModelVariant = Literal["tiny", "small", "base_plus", "large"]
DEFAULT_MODEL_VARIANT: Sam2ModelVariant = "base_plus"
SAM2_MODEL_CATALOG: dict[Sam2ModelVariant, dict[str, str | int]] = {
    "tiny": {
        "model_id": "facebook/sam2.1-hiera-tiny",
        "label": "Tiny",
        "parameters_millions": 39,
        "profile": "Fastest",
    },
    "small": {
        "model_id": "facebook/sam2.1-hiera-small",
        "label": "Small",
        "parameters_millions": 46,
        "profile": "Fast",
    },
    "base_plus": {
        "model_id": "facebook/sam2.1-hiera-base-plus",
        "label": "Base+",
        "parameters_millions": 81,
        "profile": "Balanced",
    },
    "large": {
        "model_id": "facebook/sam2.1-hiera-large",
        "label": "Large",
        "parameters_millions": 224,
        "profile": "Highest quality",
    },
}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")

for directory in (UPLOAD_ROOT, PREVIEW_ROOT, SELECTION_ROOT, OUTPUT_ROOT):
    directory.mkdir(parents=True, exist_ok=True)


class PointPayload(BaseModel):
    x: float
    y: float


class BoxPayload(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class SelectionCreateRequest(BaseModel):
    video_id: str
    start_frame: int = Field(default=0, ge=0)
    seed_frame: int | None = Field(default=None, ge=0)
    frame_count: int = Field(
        default=MAX_VIDEO_FRAMES,
        ge=MIN_VIDEO_FRAMES,
        le=MAX_VIDEO_FRAMES,
    )
    point: PointPayload | None = None
    box: BoxPayload | None = None

    @model_validator(mode="after")
    def require_prompt(self):
        if self.point is None and self.box is None:
            raise ValueError("A point or box prompt is required")
        return self


class SelectionRefineRequest(BaseModel):
    additions: list[PointPayload] = Field(default_factory=list)
    subtractions: list[PointPayload] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_correction(self):
        if not self.additions and not self.subtractions:
            raise ValueError("At least one addition or subtraction is required")
        return self


class RenderRequest(BaseModel):
    model_variant: Sam2ModelVariant = DEFAULT_MODEL_VARIANT


@dataclass(frozen=True, slots=True)
class VideoRecord:
    video_id: str
    source_path: Path
    preview_path: Path
    metadata: VideoMetadata


@dataclass(slots=True)
class SelectionRecord:
    selection_id: str
    model_session_id: str
    video_id: str
    alpha_path: Path
    start_frame: int
    seed_frame: int
    frame_count: int
    revision: int = 1
    accepted: bool = False
    accepted_mask_path: Path | None = None


@dataclass(slots=True)
class RenderJob:
    job_id: str
    selection_id: str
    video_id: str
    model_variant: Sam2ModelVariant
    start_frame: int
    seed_frame: int
    frame_count: int
    status: Literal["queued", "running", "completed", "failed"] = "queued"
    processed_frames: int = 0
    total_frames: int = MAX_VIDEO_FRAMES
    output_path: Path | None = None
    error: str | None = None


class RuntimeState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.videos: dict[str, VideoRecord] = {}
        self.selections: dict[str, SelectionRecord] = {}
        self.jobs: dict[str, RenderJob] = {}
        self.render_lock = threading.Lock()
        self.offload_state_to_cpu = _env_bool(
            "FRAMESEG_OFFLOAD_STATE_TO_CPU",
            False,
        )
        self.rotoscope_config = {
            "output_root": OUTPUT_ROOT,
            "decode_prefetch": int(os.getenv("FRAMESEG_DECODE_PREFETCH", "3")),
            "encode_queue_depth": int(os.getenv("FRAMESEG_ENCODE_QUEUE_DEPTH", "3")),
            "output_codec": os.getenv("FRAMESEG_OUTPUT_CODEC", "prores_4444"),
            "encoder_threads": int(os.getenv("FRAMESEG_ENCODER_THREADS", "0")),
            "vp9_crf": int(os.getenv("FRAMESEG_VP9_CRF", "18")),
            "vp9_cpu_used": int(os.getenv("FRAMESEG_VP9_CPU_USED", "5")),
            "reverse_decode_chunk": int(
                os.getenv("FRAMESEG_REVERSE_DECODE_CHUNK", "16")
            ),
            "keep_frame_artifacts": _env_bool(
                "FRAMESEG_KEEP_FRAME_ARTIFACTS",
                False,
            ),
        }
        self.model = Sam2ModelService(
            model_id=os.getenv(
                "FRAMESEG_MODEL_ID", "facebook/sam2.1-hiera-base-plus"
            ),
            offload_state_to_cpu=self.offload_state_to_cpu,
        )
        self.rotoscope = self.create_rotoscope(self.model)

    def create_rotoscope(self, model: Sam2ModelService) -> RotoscopeService:
        return RotoscopeService(model, **self.rotoscope_config)


state = RuntimeState()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if _env_bool("FRAMESEG_PRELOAD_MODEL", True):
        await run_in_threadpool(state.model.load)
    yield


app = FastAPI(title="FrameSeg API", version="0.4.0", lifespan=lifespan)
allowed_origins = [
    value.strip()
    for value in os.getenv("FRONTEND_ORIGINS", "http://localhost:3000").split(",")
    if value.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": state.model.is_loaded,
        "max_frames": MAX_VIDEO_FRAMES,
        "output_codec": state.rotoscope.output_codec,
    }


@app.get("/api/models")
def available_models() -> dict:
    return {
        "default": DEFAULT_MODEL_VARIANT,
        "models": [
            {"variant": variant, **metadata}
            for variant, metadata in SAM2_MODEL_CATALOG.items()
        ],
    }


@app.post("/api/videos")
async def upload_video(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4", ".mov"}:
        raise HTTPException(status_code=415, detail="Only .mp4 and .mov files are supported")

    video_id = uuid.uuid4().hex
    destination = UPLOAD_ROOT / f"{video_id}{suffix}"
    partial = destination.with_suffix(f"{suffix}.partial")
    total_bytes = 0
    try:
        with partial.open("wb") as output:
            while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Video exceeds the upload limit")
                output.write(chunk)
        partial.replace(destination)

        metadata = await run_in_threadpool(probe_video, destination)
        if _video_total_frames(metadata) < MIN_VIDEO_FRAMES:
            raise ValueError(
                f"Video must contain at least {MIN_VIDEO_FRAMES} frames"
            )
        preview_path = PREVIEW_ROOT / f"{video_id}.png"
        await run_in_threadpool(_write_video_frame, destination, preview_path, 0)
    except HTTPException:
        partial.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        raise
    except Exception as error:
        partial.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Video could not be decoded: {error}") from error
    finally:
        await file.close()

    record = VideoRecord(
        video_id=video_id,
        source_path=destination,
        preview_path=preview_path,
        metadata=metadata,
    )
    with state.lock:
        state.videos[video_id] = record
    return _video_payload(record)


@app.get("/api/videos/{video_id}/frame.png")
def video_frame(video_id: str, frame_index: int = 0) -> FileResponse:
    record = _get_video(video_id)
    _validate_clip_range(record, frame_index, 1, enforce_minimum=False)
    preview_path = _preview_path(record, frame_index)
    if not preview_path.is_file():
        _write_video_frame(record.source_path, preview_path, frame_index)
    return FileResponse(
        preview_path,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/api/videos/{video_id}/source")
def source_video(video_id: str) -> FileResponse:
    record = _get_video(video_id)
    media_type = (
        "video/quicktime" if record.source_path.suffix.lower() == ".mov" else "video/mp4"
    )
    return FileResponse(record.source_path, media_type=media_type)


@app.delete("/api/videos/{video_id}", status_code=204)
def delete_video(video_id: str) -> Response:
    """Discard one video's source, previews, masks, renders, and runtime state."""
    video = _get_video(video_id)
    with state.lock:
        jobs = [job for job in state.jobs.values() if job.video_id == video_id]
        if any(job.status in {"queued", "running"} for job in jobs):
            raise HTTPException(
                status_code=409,
                detail="Wait for the active render to finish before discarding this video",
            )
        selections = [
            selection
            for selection in state.selections.values()
            if selection.video_id == video_id
        ]

    for selection in selections:
        if not selection.accepted:
            state.rotoscope.cancel_selection(selection.model_session_id)

    video.source_path.unlink(missing_ok=True)
    for preview_path in PREVIEW_ROOT.glob(f"{video_id}*.png"):
        preview_path.unlink(missing_ok=True)
    for selection in selections:
        shutil.rmtree(SELECTION_ROOT / selection.selection_id, ignore_errors=True)
    for job in jobs:
        shutil.rmtree(OUTPUT_ROOT / job.job_id, ignore_errors=True)

    with state.lock:
        state.videos.pop(video_id, None)
        for selection in selections:
            state.selections.pop(selection.selection_id, None)
        for job in jobs:
            state.jobs.pop(job.job_id, None)

    return Response(status_code=204)


@app.post("/api/selections")
async def create_selection(request: SelectionCreateRequest) -> dict:
    video = _get_video(request.video_id)
    _validate_clip_range(video, request.start_frame, request.frame_count)
    seed_frame = (
        request.start_frame if request.seed_frame is None else request.seed_frame
    )
    _validate_seed_frame(request.start_frame, request.frame_count, seed_frame)
    if request.point is not None:
        prompt = ObjectPrompt.positive_point(request.point.x, request.point.y)
    else:
        assert request.box is not None
        prompt = ObjectPrompt.bounding_box(
            request.box.x1,
            request.box.y1,
            request.box.x2,
            request.box.y2,
        )

    try:
        preview_path = _preview_path(video, seed_frame)
        if not preview_path.is_file():
            await run_in_threadpool(
                _write_video_frame,
                video.source_path,
                preview_path,
                seed_frame,
            )
        frame_rgb = await run_in_threadpool(_load_rgb_png, preview_path)
        preview = await run_in_threadpool(
            state.rotoscope.create_selection_preview,
            video.source_path,
            prompt,
            "auto",
            frame_rgb,
            video.metadata,
            seed_frame,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Object selection failed: {error}") from error

    selection_id = uuid.uuid4().hex
    selection_dir = SELECTION_ROOT / selection_id
    selection_dir.mkdir(parents=True, exist_ok=True)
    alpha_path = selection_dir / "alpha.png"
    try:
        _save_grayscale_png(preview.alpha, alpha_path)
    except Exception:
        state.model.close_selection(preview.session_id)
        raise

    record = SelectionRecord(
        selection_id=selection_id,
        model_session_id=preview.session_id,
        video_id=video.video_id,
        alpha_path=alpha_path,
        start_frame=request.start_frame,
        seed_frame=seed_frame,
        frame_count=request.frame_count,
    )
    with state.lock:
        state.selections[selection_id] = record
    return _selection_payload(record, preview.selection.object_score)


@app.patch("/api/selections/{selection_id}")
async def refine_selection(selection_id: str, request: SelectionRefineRequest) -> dict:
    record = _get_selection(selection_id)
    if record.accepted:
        raise HTTPException(status_code=409, detail="Selection has already been accepted")

    additions = tuple((point.x, point.y) for point in request.additions)
    subtractions = tuple((point.x, point.y) for point in request.subtractions)
    try:
        update = await run_in_threadpool(
            state.rotoscope.refine_selection_preview,
            record.model_session_id,
            additions,
            subtractions,
        )
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Mask refinement failed: {error}") from error

    _save_grayscale_png(update.alpha, record.alpha_path)
    with state.lock:
        record.revision += 1
    return _selection_payload(record, update.selection.object_score)


@app.get("/api/selections/{selection_id}/alpha.png")
def selection_alpha(selection_id: str) -> FileResponse:
    record = _get_selection(selection_id)
    return FileResponse(
        record.alpha_path,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


@app.delete("/api/selections/{selection_id}")
def cancel_selection(selection_id: str) -> dict:
    record = _get_selection(selection_id)
    if not record.accepted:
        state.rotoscope.cancel_selection(record.model_session_id)
    with state.lock:
        state.selections.pop(selection_id, None)
    return {"status": "cancelled"}


@app.post("/api/selections/{selection_id}/render", status_code=202)
def start_render(
    selection_id: str,
    request: RenderRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    selection = _get_selection(selection_id)
    video = _get_video(selection.video_id)

    with state.lock:
        already_running = any(
            existing.selection_id == selection_id
            and existing.status in {"queued", "running"}
            for existing in state.jobs.values()
        )
        if already_running:
            raise HTTPException(
                status_code=409,
                detail="A render for this selection is already running",
            )

        if not selection.accepted:
            accepted = state.rotoscope.accept_selection(selection.model_session_id)
            accepted_path = selection.alpha_path.parent / "accepted_mask.png"
            _save_grayscale_png(
                accepted.binary_mask.astype(np.uint8) * 255,
                accepted_path,
            )
            selection.accepted = True
            selection.accepted_mask_path = accepted_path
        else:
            accepted_path = selection.accepted_mask_path
            if accepted_path is None or not accepted_path.is_file():
                raise HTTPException(
                    status_code=409,
                    detail="The accepted mask is unavailable; create the selection again",
                )

    job_id = uuid.uuid4().hex
    job = RenderJob(
        job_id=job_id,
        selection_id=selection_id,
        video_id=video.video_id,
        model_variant=request.model_variant,
        start_frame=selection.start_frame,
        seed_frame=selection.seed_frame,
        frame_count=selection.frame_count,
        total_frames=selection.frame_count,
    )
    with state.lock:
        state.jobs[job_id] = job
    background_tasks.add_task(
        _run_render_job,
        job_id,
        video.source_path,
        accepted_path,
        selection.start_frame,
        selection.seed_frame,
        selection.frame_count,
        request.model_variant,
    )
    return _job_payload(job)


@app.get("/api/jobs/{job_id}")
def render_status(job_id: str) -> dict:
    return _job_payload(_get_job(job_id))


@app.get("/api/jobs/{job_id}/output")
def download_output(job_id: str) -> FileResponse:
    job = _get_job(job_id)
    if job.status != "completed" or job.output_path is None:
        raise HTTPException(status_code=409, detail="Render output is not ready")
    suffix = job.output_path.suffix.lower()
    return FileResponse(
        job.output_path,
        media_type="video/webm" if suffix == ".webm" else "video/quicktime",
        filename=f"frameseg-{job_id}{suffix}",
    )


def _run_render_job(
    job_id: str,
    video_path: Path,
    accepted_mask_path: Path,
    start_frame: int,
    seed_frame: int,
    frame_count: int,
    model_variant: Sam2ModelVariant,
) -> None:
    job = _get_job(job_id)
    render_model: Sam2ModelService | None = None
    try:
        with Image.open(accepted_mask_path) as image:
            accepted_mask = np.asarray(image.convert("L"), dtype=np.uint8) > 127

        def update_progress(processed_frames: int) -> None:
            with state.lock:
                job.processed_frames = processed_frames

        model_id = str(SAM2_MODEL_CATALOG[model_variant]["model_id"])
        with state.render_lock:
            with state.lock:
                job.status = "running"
            try:
                if model_id == state.model.model_id:
                    rotoscope = state.rotoscope
                else:
                    render_model = Sam2ModelService(
                        model_id=model_id,
                        offload_state_to_cpu=state.offload_state_to_cpu,
                    )
                    rotoscope = state.create_rotoscope(render_model)

                result = rotoscope.render(
                    video_path,
                    accepted_mask=accepted_mask,
                    start_frame=start_frame,
                    max_frames=frame_count,
                    seed_frame=seed_frame,
                    job_id=job_id,
                    on_progress=update_progress,
                )
            finally:
                if render_model is not None:
                    render_model.unload()
                    render_model = None

            with state.lock:
                job.status = "completed"
                job.processed_frames = result.output.frame_count
                job.output_path = result.output.video_path
    except Exception as error:
        with state.lock:
            job.status = "failed"
            job.error = str(error)


def _write_video_frame(
    video_path: Path,
    destination: Path,
    frame_index: int,
) -> None:
    packets = iter_video_frames(
        video_path,
        max_frames=1,
        start_frame=frame_index,
    )
    try:
        packet = next(packets)
    finally:
        packets.close()
    _save_rgb_png(packet.rgb, destination)


def _save_rgb_png(rgb: np.ndarray, destination: Path) -> None:
    _save_image_atomic(Image.fromarray(np.asarray(rgb, dtype=np.uint8)), destination)


def _load_rgb_png(source: Path) -> np.ndarray:
    with Image.open(source) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _save_grayscale_png(alpha: np.ndarray, destination: Path) -> None:
    _save_image_atomic(Image.fromarray(np.asarray(alpha, dtype=np.uint8)), destination)


def _save_image_atomic(image: Image.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.partial"
    )
    try:
        image.save(temporary, format="PNG")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _get_video(video_id: str) -> VideoRecord:
    with state.lock:
        record = state.videos.get(video_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return record


def _video_total_frames(metadata: VideoMetadata) -> int:
    if metadata.total_frames is not None and metadata.total_frames > 0:
        return metadata.total_frames
    if metadata.duration_seconds is not None:
        return max(1, int(round(metadata.duration_seconds * metadata.fps)))
    raise ValueError("Video frame count could not be determined")


def _validate_clip_range(
    video: VideoRecord,
    start_frame: int,
    frame_count: int,
    *,
    enforce_minimum: bool = True,
) -> None:
    total_frames = _video_total_frames(video.metadata)
    if start_frame < 0 or start_frame >= total_frames:
        raise HTTPException(status_code=422, detail="Start frame is outside the video")
    if enforce_minimum and frame_count < MIN_VIDEO_FRAMES:
        raise HTTPException(
            status_code=422,
            detail=f"Select at least {MIN_VIDEO_FRAMES} frames",
        )
    if frame_count > MAX_VIDEO_FRAMES:
        raise HTTPException(
            status_code=422,
            detail=f"Select no more than {MAX_VIDEO_FRAMES} frames",
        )
    if start_frame + frame_count > total_frames:
        raise HTTPException(
            status_code=422,
            detail="Selected frame range extends beyond the video",
        )


def _validate_seed_frame(
    start_frame: int,
    frame_count: int,
    seed_frame: int,
) -> None:
    if not start_frame <= seed_frame < start_frame + frame_count:
        raise HTTPException(
            status_code=422,
            detail="Selection frame must be inside the selected clip range",
        )


def _preview_path(video: VideoRecord, frame_index: int) -> Path:
    if frame_index == 0:
        return video.preview_path
    return PREVIEW_ROOT / f"{video.video_id}-{frame_index:09d}.png"


def _get_selection(selection_id: str) -> SelectionRecord:
    with state.lock:
        record = state.selections.get(selection_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Selection not found")
    return record


def _get_job(job_id: str) -> RenderJob:
    with state.lock:
        job = state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Render job not found")
    return job


def _video_payload(record: VideoRecord) -> dict:
    metadata = record.metadata
    total_frames = _video_total_frames(metadata)
    return {
        "video_id": record.video_id,
        "first_frame_url": f"/api/videos/{record.video_id}/frame.png",
        "source_url": f"/api/videos/{record.video_id}/source",
        "metadata": {
            "container": metadata.container,
            "codec": metadata.codec,
            "width": metadata.width,
            "height": metadata.height,
            "fps": metadata.fps,
            "duration_seconds": metadata.duration_seconds,
            "total_frames": total_frames,
            "min_clip_frames": MIN_VIDEO_FRAMES,
            "max_frames": MAX_VIDEO_FRAMES,
        },
    }


def _selection_payload(record: SelectionRecord, object_score: float) -> dict:
    return {
        "selection_id": record.selection_id,
        "alpha_url": f"/api/selections/{record.selection_id}/alpha.png?v={record.revision}",
        "revision": record.revision,
        "object_score": object_score,
        "start_frame": record.start_frame,
        "seed_frame": record.seed_frame,
        "frame_count": record.frame_count,
    }


def _job_payload(job: RenderJob) -> dict:
    progress = job.processed_frames / job.total_frames if job.total_frames else 0.0
    return {
        "job_id": job.job_id,
        "model_variant": job.model_variant,
        "start_frame": job.start_frame,
        "seed_frame": job.seed_frame,
        "frame_count": job.frame_count,
        "status": job.status,
        "processed_frames": job.processed_frames,
        "total_frames": job.total_frames,
        "progress": min(progress, 1.0),
        "error": job.error,
        "output_url": f"/api/jobs/{job.job_id}/output" if job.status == "completed" else None,
    }


__all__ = ["app"]
