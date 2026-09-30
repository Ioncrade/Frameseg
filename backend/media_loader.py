from __future__ import annotations

import json
import queue
import shutil
import subprocess
import threading
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from collections.abc import Iterator
from typing import Literal

import av
import numpy as np


MAX_VIDEO_FRAMES = 1_200
MIN_VIDEO_FRAMES = 30
SUPPORTED_CONTAINERS = frozenset({".mp4", ".mov"})
SUPPORTED_CODECS = {
    "avc1": "h264",
    "h264": "h264",
    "h265": "h265",
    "hevc": "h265",
    "hev1": "h265",
    "hvc1": "h265",
}

Decoder = Literal["auto", "pyav", "ffmpeg"]


class MediaLoaderError(RuntimeError):
    """Base error raised by the media loader."""


class UnsupportedMediaError(MediaLoaderError):
    """Raised when the container or video codec is unsupported."""


class VideoDecodeError(MediaLoaderError):
    """Raised when neither decoder can read the video."""


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    path: Path
    container: str
    codec: str
    width: int
    height: int
    fps: float
    duration_seconds: float | None
    total_frames: int | None
    decoded_frames: int
    decoder: Literal["pyav", "ffmpeg"]


@dataclass(slots=True)
class LoadedVideo:
    frames: list[np.ndarray]
    metadata: VideoMetadata


@dataclass(frozen=True, slots=True)
class FramePacket:
    index: int
    rgb: np.ndarray


@dataclass(frozen=True, slots=True)
class _PrefetchFailure:
    error: BaseException


_PREFETCH_END = object()


@dataclass(frozen=True, slots=True)
class _VideoProbe:
    codec: str
    width: int
    height: int
    fps: float
    duration_seconds: float | None
    total_frames: int | None


def load_video(
    video_path: str | Path,
    max_frames: int = MAX_VIDEO_FRAMES,
    decoder: Decoder = "auto",
    start_frame: int = 0,
) -> LoadedVideo:
    """Decode a bounded consecutive frame range into memory.

    Supported inputs are H.264 and H.265/HEVC video in MP4 or MOV containers.
    PyAV (FFmpeg bindings) is preferred. With ``decoder="auto"``, the FFmpeg
    command-line decoder is used if PyAV cannot open or decode the video.
    """
    path = _validate_request(video_path, max_frames, decoder, start_frame)
    frame_limit = min(max_frames, MAX_VIDEO_FRAMES)

    if decoder in {"auto", "pyav"}:
        try:
            frames, probe = _decode_with_pyav(path, frame_limit, start_frame)
            return _build_result(path, frames, probe, "pyav")
        except UnsupportedMediaError:
            raise
        except Exception as error:
            pyav_error = error
            if decoder == "pyav":
                raise VideoDecodeError(f"PyAV could not decode {path}: {error}") from error
    else:
        pyav_error = None

    try:
        frames, probe = _decode_with_ffmpeg(path, frame_limit, start_frame)
        return _build_result(path, frames, probe, "ffmpeg")
    except UnsupportedMediaError:
        raise
    except Exception as error:
        if pyav_error is None:
            raise VideoDecodeError(f"FFmpeg could not decode {path}: {error}") from error
        raise VideoDecodeError(
            f"Both decoders failed for {path}. "
            f"PyAV: {pyav_error}. FFmpeg: {error}"
        ) from error


def probe_video(
    video_path: str | Path,
    decoder: Decoder = "auto",
) -> VideoMetadata:
    """Read video metadata without materializing its frames."""
    path = _validate_request(video_path, 1, decoder, 0)

    if decoder in {"auto", "pyav"}:
        try:
            with av.open(str(path), mode="r") as container:
                stream = next(
                    (item for item in container.streams if item.type == "video"), None
                )
                if stream is None:
                    raise VideoDecodeError("The file does not contain a video stream")
                probe = _probe_pyav(container, stream)
                _require_supported_codec(probe.codec)
                return _build_metadata(path, probe, "pyav", decoded_frames=0)
        except UnsupportedMediaError:
            raise
        except Exception as error:
            if decoder == "pyav":
                raise VideoDecodeError(f"PyAV could not probe {path}: {error}") from error

    probe = _probe_ffmpeg(path)
    _require_supported_codec(probe.codec)
    return _build_metadata(path, probe, "ffmpeg", decoded_frames=0)


def iter_video_frames(
    video_path: str | Path,
    max_frames: int = MAX_VIDEO_FRAMES,
    decoder: Decoder = "auto",
    start_frame: int = 0,
) -> Iterator[FramePacket]:
    """Stream a bounded frame range without retaining prior frames in memory.

    In automatic mode, FFmpeg is used if PyAV fails before producing a frame. A
    mid-stream PyAV failure is surfaced instead of silently restarting the clip
    and yielding duplicate frame indices.
    """
    path = _validate_request(video_path, max_frames, decoder, start_frame)
    frame_limit = min(max_frames, MAX_VIDEO_FRAMES)

    if decoder in {"auto", "pyav"}:
        yielded_frames = 0
        try:
            for packet in _iter_frames_pyav(path, frame_limit, start_frame):
                yielded_frames += 1
                yield packet
            return
        except UnsupportedMediaError:
            raise
        except Exception as error:
            if decoder == "pyav" or yielded_frames:
                raise VideoDecodeError(
                    f"PyAV failed after {yielded_frames} decoded frame(s): {error}"
                ) from error

    yield from _iter_frames_ffmpeg(path, frame_limit, start_frame)


def prefetch_video_frames(
    frames: Iterator[FramePacket],
    buffer_size: int = 3,
) -> Iterator[FramePacket]:
    """Decode ahead on one thread while keeping a strict memory bound.

    A small queue overlaps CPU video decode with GPU inference without ever
    materializing the whole clip. Closing the returned iterator also asks the
    producer to stop and closes the source iterator on the producer thread.
    """
    if (
        isinstance(buffer_size, bool)
        or not isinstance(buffer_size, int)
        or buffer_size < 1
    ):
        raise ValueError("buffer_size must be a positive integer")

    items: queue.Queue[FramePacket | _PrefetchFailure | object] = queue.Queue(
        maxsize=buffer_size
    )
    stop = threading.Event()

    def enqueue(item: FramePacket | _PrefetchFailure | object) -> bool:
        while not stop.is_set():
            try:
                items.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def produce() -> None:
        try:
            for packet in frames:
                if not enqueue(packet):
                    break
        except BaseException as error:
            enqueue(_PrefetchFailure(error))
        finally:
            close_source = getattr(frames, "close", None)
            if close_source is not None:
                close_source()
            enqueue(_PREFETCH_END)

    producer = threading.Thread(
        target=produce,
        name="frameseg-video-prefetch",
        daemon=True,
    )
    producer.start()
    try:
        while True:
            item = items.get()
            if item is _PREFETCH_END:
                return
            if isinstance(item, _PrefetchFailure):
                raise item.error
            assert isinstance(item, FramePacket)
            yield item
    finally:
        stop.set()
        producer.join(timeout=5.0)


def _validate_request(
    video_path: str | Path,
    max_frames: int,
    decoder: Decoder,
    start_frame: int,
) -> Path:
    path = Path(video_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    if path.suffix.lower() not in SUPPORTED_CONTAINERS:
        supported = ", ".join(sorted(SUPPORTED_CONTAINERS))
        raise UnsupportedMediaError(
            f"Unsupported container {path.suffix!r}; expected one of: {supported}"
        )
    if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1:
        raise ValueError("max_frames must be a positive integer")
    if isinstance(start_frame, bool) or not isinstance(start_frame, int) or start_frame < 0:
        raise ValueError("start_frame must be a non-negative integer")
    if decoder not in {"auto", "pyav", "ffmpeg"}:
        raise ValueError("decoder must be 'auto', 'pyav', or 'ffmpeg'")
    return path


def _decode_with_pyav(
    path: Path,
    frame_limit: int,
    start_frame: int,
) -> tuple[list[np.ndarray], _VideoProbe]:
    frames: list[np.ndarray] = []
    with av.open(str(path), mode="r") as container:
        stream = next((item for item in container.streams if item.type == "video"), None)
        if stream is None:
            raise VideoDecodeError("The file does not contain a video stream")

        probe = _probe_pyav(container, stream)
        _require_supported_codec(probe.codec)
        stream.thread_type = "AUTO"
        target_seconds = _seek_pyav(container, stream, start_frame, probe.fps)

        for frame in container.decode(stream):
            if _frame_precedes_target(frame, stream, target_seconds, probe.fps):
                continue
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= frame_limit:
                break

    if not frames:
        raise VideoDecodeError("PyAV returned no decoded video frames")
    return frames, probe


def _iter_frames_pyav(
    path: Path,
    frame_limit: int,
    start_frame: int,
) -> Iterator[FramePacket]:
    with av.open(str(path), mode="r") as container:
        stream = next((item for item in container.streams if item.type == "video"), None)
        if stream is None:
            raise VideoDecodeError("The file does not contain a video stream")
        probe = _probe_pyav(container, stream)
        _require_supported_codec(probe.codec)
        stream.thread_type = "AUTO"
        target_seconds = _seek_pyav(container, stream, start_frame, probe.fps)

        decoded_frames = 0
        for frame in container.decode(stream):
            if _frame_precedes_target(frame, stream, target_seconds, probe.fps):
                continue
            if decoded_frames >= frame_limit:
                break
            yield FramePacket(
                index=decoded_frames,
                rgb=frame.to_ndarray(format="rgb24"),
            )
            decoded_frames += 1

    if decoded_frames == 0:
        raise VideoDecodeError("PyAV returned no decoded video frames")


def _probe_pyav(container: av.container.InputContainer, stream) -> _VideoProbe:
    codec_name = stream.codec_context.name or "unknown"
    fps = float(stream.average_rate) if stream.average_rate else 0.0
    total_frames = int(stream.frames) if stream.frames else None

    duration_seconds: float | None = None
    if stream.duration is not None and stream.time_base is not None:
        duration_seconds = float(stream.duration * stream.time_base)
    elif container.duration is not None:
        duration_seconds = float(container.duration / av.time_base)

    if fps <= 0:
        raise VideoDecodeError("The video stream has no valid frame rate")

    return _VideoProbe(
        codec=_normalize_codec(codec_name),
        width=int(stream.codec_context.width),
        height=int(stream.codec_context.height),
        fps=fps,
        duration_seconds=duration_seconds,
        total_frames=total_frames,
    )


def _decode_with_ffmpeg(
    path: Path,
    frame_limit: int,
    start_frame: int,
) -> tuple[list[np.ndarray], _VideoProbe]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise VideoDecodeError("The ffmpeg executable is not installed or not on PATH")

    probe = _probe_ffmpeg(path)
    _require_supported_codec(probe.codec)
    frame_size = probe.width * probe.height * 3
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-ss",
        _frame_timestamp(start_frame, probe.fps),
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-frames:v",
        str(frame_limit),
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdout is None or process.stderr is None:
        process.kill()
        raise VideoDecodeError("Failed to open FFmpeg output pipes")

    frames: list[np.ndarray] = []
    try:
        for _ in range(frame_limit):
            raw_frame = process.stdout.read(frame_size)
            if not raw_frame:
                break
            if len(raw_frame) != frame_size:
                raise VideoDecodeError(
                    f"FFmpeg returned a partial frame ({len(raw_frame)} of {frame_size} bytes)"
                )
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                probe.height, probe.width, 3
            )
            frames.append(frame)

        process.stdout.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
        return_code = process.wait()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    if return_code != 0:
        raise VideoDecodeError(stderr or f"FFmpeg exited with status {return_code}")
    if not frames:
        raise VideoDecodeError(stderr or "FFmpeg returned no decoded video frames")
    return frames, probe


def _iter_frames_ffmpeg(
    path: Path,
    frame_limit: int,
    start_frame: int,
) -> Iterator[FramePacket]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise VideoDecodeError("The ffmpeg executable is not installed or not on PATH")

    probe = _probe_ffmpeg(path)
    _require_supported_codec(probe.codec)
    frame_size = probe.width * probe.height * 3
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-ss",
        _frame_timestamp(start_frame, probe.fps),
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-frames:v",
        str(frame_limit),
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdout is None or process.stderr is None:
        process.kill()
        raise VideoDecodeError("Failed to open FFmpeg output pipes")

    decoded_frames = 0
    try:
        for frame_index in range(frame_limit):
            raw_frame = process.stdout.read(frame_size)
            if not raw_frame:
                break
            if len(raw_frame) != frame_size:
                raise VideoDecodeError(
                    f"FFmpeg returned a partial frame ({len(raw_frame)} of {frame_size} bytes)"
                )
            decoded_frames += 1
            yield FramePacket(
                index=frame_index,
                rgb=np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                    probe.height, probe.width, 3
                ),
            )

        process.stdout.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
        return_code = process.wait()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    if return_code != 0:
        raise VideoDecodeError(stderr or f"FFmpeg exited with status {return_code}")
    if decoded_frames == 0:
        raise VideoDecodeError(stderr or "FFmpeg returned no decoded video frames")


def _probe_ffmpeg(path: Path) -> _VideoProbe:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise VideoDecodeError("The ffprobe executable is not installed or not on PATH")

    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,nb_frames,duration:format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise VideoDecodeError(result.stderr.strip() or "ffprobe failed")

    try:
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]
    except (IndexError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise VideoDecodeError("ffprobe returned invalid video metadata") from error

    fps = _parse_frame_rate(stream.get("avg_frame_rate"))
    if fps <= 0:
        raise VideoDecodeError("The video stream has no valid frame rate")

    duration_value = stream.get("duration") or payload.get("format", {}).get("duration")
    duration = _optional_float(duration_value)
    total_frames = _optional_int(stream.get("nb_frames"))

    return _VideoProbe(
        codec=_normalize_codec(str(stream.get("codec_name", "unknown"))),
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=fps,
        duration_seconds=duration,
        total_frames=total_frames,
    )


def _seek_pyav(container, stream, start_frame: int, fps: float) -> float:
    """Seek to the keyframe before a target and return its target timestamp."""
    target_seconds = start_frame / fps
    if start_frame and stream.time_base is not None:
        target_pts = int(target_seconds / float(stream.time_base))
        container.seek(max(target_pts, 0), backward=True, any_frame=False, stream=stream)
    return target_seconds


def _frame_precedes_target(frame, stream, target_seconds: float, fps: float) -> bool:
    if target_seconds <= 0 or frame.pts is None or stream.time_base is None:
        return False
    frame_seconds = float(frame.pts * stream.time_base)
    return frame_seconds < target_seconds - (0.5 / fps)


def _frame_timestamp(frame_index: int, fps: float) -> str:
    # Aim at the middle of the preceding frame. Some MP4 streams have a small
    # presentation-time offset, and seeking to the exact boundary can otherwise
    # make FFmpeg return the following frame.
    return f"{max(0.0, (frame_index - 0.5) / fps):.9f}"


def _build_result(
    path: Path,
    frames: list[np.ndarray],
    probe: _VideoProbe,
    decoder: Literal["pyav", "ffmpeg"],
) -> LoadedVideo:
    metadata = _build_metadata(path, probe, decoder, decoded_frames=len(frames))
    return LoadedVideo(frames=frames, metadata=metadata)


def _build_metadata(
    path: Path,
    probe: _VideoProbe,
    decoder: Literal["pyav", "ffmpeg"],
    decoded_frames: int,
) -> VideoMetadata:
    return VideoMetadata(
        path=path,
        container=path.suffix.lower().lstrip("."),
        codec=probe.codec,
        width=probe.width,
        height=probe.height,
        fps=probe.fps,
        duration_seconds=probe.duration_seconds,
        total_frames=probe.total_frames,
        decoded_frames=decoded_frames,
        decoder=decoder,
    )


def _normalize_codec(codec_name: str) -> str:
    normalized = codec_name.strip().lower()
    return SUPPORTED_CODECS.get(normalized, normalized)


def _require_supported_codec(codec_name: str) -> None:
    if codec_name not in {"h264", "h265"}:
        raise UnsupportedMediaError(
            f"Unsupported video codec {codec_name!r}; expected H.264 or H.265/HEVC"
        )


def _parse_frame_rate(value: object) -> float:
    if not value or value == "0/0":
        return 0.0
    try:
        return float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _optional_float(value: object) -> float | None:
    try:
        return float(value) if value not in {None, "N/A"} else None
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    try:
        return int(value) if value not in {None, "N/A"} else None
    except (TypeError, ValueError):
        return None


__all__ = [
    "LoadedVideo",
    "FramePacket",
    "MAX_VIDEO_FRAMES",
    "MIN_VIDEO_FRAMES",
    "MediaLoaderError",
    "UnsupportedMediaError",
    "VideoDecodeError",
    "VideoMetadata",
    "iter_video_frames",
    "load_video",
    "prefetch_video_frames",
    "probe_video",
]
