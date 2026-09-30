from __future__ import annotations

import re
import shutil
import uuid
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from mask_service import MaskOutputService, OutputCodec, OutputManifest, mask_logits_to_alpha
from media_loader import (
    Decoder,
    FramePacket,
    MAX_VIDEO_FRAMES,
    VideoMetadata,
    iter_video_frames,
    prefetch_video_frames,
    probe_video,
)
from model_service import ObjectPrompt, ObjectSelection, Sam2ModelService


_SAFE_JOB_ID = re.compile(r"^[A-Za-z0-9_-]{1,100}$")


@dataclass(frozen=True, slots=True)
class SelectionPreview:
    session_id: str
    frame_rgb: np.ndarray
    alpha: np.ndarray
    selection: ObjectSelection
    video: VideoMetadata


@dataclass(frozen=True, slots=True)
class SelectionUpdate:
    session_id: str
    alpha: np.ndarray
    selection: ObjectSelection


@dataclass(frozen=True, slots=True)
class RotoscopeResult:
    job_id: str
    video: VideoMetadata
    output: OutputManifest


class RotoscopeService:
    """Orchestrate selection preview and bounded-memory rotoscoping jobs."""

    def __init__(
        self,
        model_service: Sam2ModelService,
        output_root: str | Path = "outputs",
        decode_prefetch: int = 3,
        encode_queue_depth: int = 3,
        output_codec: OutputCodec = "prores_4444",
        encoder_threads: int = 0,
        vp9_crf: int = 18,
        vp9_cpu_used: int = 5,
        reverse_decode_chunk: int = 16,
        keep_frame_artifacts: bool = False,
    ) -> None:
        if decode_prefetch < 0:
            raise ValueError("decode_prefetch cannot be negative")
        if encode_queue_depth < 1:
            raise ValueError("encode_queue_depth must be positive")
        if output_codec not in {"vp9_alpha", "qtrle", "prores_4444"}:
            raise ValueError(
                "output_codec must be 'vp9_alpha', 'qtrle', or 'prores_4444'"
            )
        if encoder_threads < 0:
            raise ValueError("encoder_threads cannot be negative")
        if isinstance(vp9_crf, bool) or not 0 <= vp9_crf <= 63:
            raise ValueError("vp9_crf must be between 0 and 63")
        if isinstance(vp9_cpu_used, bool) or not 0 <= vp9_cpu_used <= 8:
            raise ValueError("vp9_cpu_used must be between 0 and 8")
        if (
            isinstance(reverse_decode_chunk, bool)
            or not 1 <= reverse_decode_chunk <= 64
        ):
            raise ValueError("reverse_decode_chunk must be between 1 and 64")
        self.model_service = model_service
        self.output_root = Path(output_root)
        self.decode_prefetch = decode_prefetch
        self.encode_queue_depth = encode_queue_depth
        self.output_codec = output_codec
        self.encoder_threads = encoder_threads
        self.vp9_crf = vp9_crf
        self.vp9_cpu_used = vp9_cpu_used
        self.reverse_decode_chunk = reverse_decode_chunk
        self.keep_frame_artifacts = keep_frame_artifacts
        self.output_root.mkdir(parents=True, exist_ok=True)

    def create_selection_preview(
        self,
        video_path: str | Path,
        prompt: ObjectPrompt,
        decoder: Decoder = "auto",
        frame_rgb: np.ndarray | None = None,
        video_metadata: VideoMetadata | None = None,
        start_frame: int = 0,
    ) -> SelectionPreview:
        """Return SAM2's full-object selection on the chosen seed frame."""
        video = video_metadata or probe_video(video_path, decoder=decoder)
        if frame_rgb is None:
            frame_stream = iter_video_frames(
                video_path,
                max_frames=1,
                decoder=decoder,
                start_frame=start_frame,
            )
            try:
                first_frame_rgb = next(frame_stream).rgb
            finally:
                frame_stream.close()
        else:
            first_frame_rgb = np.asarray(frame_rgb, dtype=np.uint8)

        result = self.model_service.start_selection(first_frame_rgb, prompt)
        return SelectionPreview(
            session_id=result.session_id,
            frame_rgb=first_frame_rgb,
            alpha=mask_logits_to_alpha(result.selection.mask_logits),
            selection=result.selection,
            video=video,
        )

    def refine_selection_preview(
        self,
        session_id: str,
        additions: tuple[tuple[float, float], ...] = (),
        subtractions: tuple[tuple[float, float], ...] = (),
    ) -> SelectionUpdate:
        """Add or subtract regions and return the updated full-mask overlay."""
        selection = self.model_service.refine_selection(
            session_id,
            additions=additions,
            subtractions=subtractions,
        )
        return SelectionUpdate(
            session_id=session_id,
            alpha=mask_logits_to_alpha(selection.mask_logits),
            selection=selection,
        )

    def accept_selection(self, session_id: str) -> ObjectSelection:
        """Return the approved mask and release its interactive model session."""
        selection = self.model_service.get_selection(session_id)
        self.model_service.close_selection(session_id)
        return selection

    def cancel_selection(self, session_id: str) -> None:
        self.model_service.close_selection(session_id)

    def render(
        self,
        video_path: str | Path,
        accepted_mask: np.ndarray,
        object_id: int = 1,
        start_frame: int = 0,
        max_frames: int = MAX_VIDEO_FRAMES,
        seed_frame: int | None = None,
        decoder: Decoder = "auto",
        job_id: str | None = None,
        on_progress: Callable[[int], None] | None = None,
    ) -> RotoscopeResult:
        """Track from a seed in both directions and stream a chronological export."""
        resolved_job_id = job_id or uuid.uuid4().hex
        if not _SAFE_JOB_ID.fullmatch(resolved_job_id):
            raise ValueError("job_id may contain only letters, numbers, '_' and '-'")

        resolved_seed_frame = start_frame if seed_frame is None else seed_frame
        end_frame = start_frame + max_frames
        if not start_frame <= resolved_seed_frame < end_frame:
            raise ValueError("seed_frame must be inside the selected frame range")
        seed_offset = resolved_seed_frame - start_frame

        video = probe_video(video_path, decoder=decoder)
        writer = MaskOutputService(
            output_dir=self.output_root / resolved_job_id,
            fps=video.fps,
            keep_frame_artifacts=self.keep_frame_artifacts,
            output_codec=self.output_codec,
            encoder_threads=self.encoder_threads,
            vp9_crf=self.vp9_crf,
            vp9_cpu_used=self.vp9_cpu_used,
        )
        reverse_masks_dir = writer.output_dir / ".reverse_masks"
        shutil.rmtree(reverse_masks_dir, ignore_errors=True)
        encoder = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"frameseg-encode-{resolved_job_id[:8]}",
        )
        pending: deque[tuple[int, Future]] = deque()

        def finish_oldest() -> None:
            frame_index, future = pending.popleft()
            future.result()
            if on_progress is not None:
                on_progress(frame_index + 1)

        def submit_frame(frame_index: int, rgb: np.ndarray, alpha: np.ndarray) -> None:
            pending.append(
                (
                    frame_index,
                    encoder.submit(
                        writer.write_frame,
                        frame_index=frame_index,
                        rgb=rgb,
                        alpha=alpha,
                    ),
                )
            )
            if len(pending) >= self.encode_queue_depth:
                finish_oldest()

        try:
            if seed_offset:
                reverse_masks_dir.mkdir(parents=True, exist_ok=True)
                reverse_packets = _iter_reverse_clip_frames(
                    video_path,
                    clip_start=start_frame,
                    seed_offset=seed_offset,
                    chunk_size=self.reverse_decode_chunk,
                    decoder=decoder,
                )
                reverse_tracking = self.model_service.track_frames(
                    reverse_packets,
                    initial_mask=accepted_mask,
                    object_id=object_id,
                    reverse=True,
                )
                try:
                    for tracked in reverse_tracking:
                        if tracked.frame_index < seed_offset:
                            _save_alpha_mask(
                                tracked.alpha,
                                reverse_masks_dir / _mask_filename(tracked.frame_index),
                            )
                finally:
                    reverse_tracking.close()
                    reverse_packets.close()

                prefix_packets = iter_video_frames(
                    video_path,
                    max_frames=seed_offset,
                    decoder=decoder,
                    start_frame=start_frame,
                )
                try:
                    for packet in prefix_packets:
                        mask_path = reverse_masks_dir / _mask_filename(packet.index)
                        with Image.open(mask_path) as image:
                            alpha = np.asarray(image.convert("L"), dtype=np.uint8).copy()
                        submit_frame(packet.index, packet.rgb, alpha)
                finally:
                    prefix_packets.close()

            forward_source = iter_video_frames(
                video_path,
                max_frames=max_frames - seed_offset,
                decoder=decoder,
                start_frame=resolved_seed_frame,
            )
            forward_packets = _reindex_packets(forward_source, seed_offset)
            packets = (
                prefetch_video_frames(forward_packets, buffer_size=self.decode_prefetch)
                if self.decode_prefetch
                else forward_packets
            )
            forward_tracking = self.model_service.track_frames(
                packets,
                initial_mask=accepted_mask,
                object_id=object_id,
            )
            try:
                for tracked in forward_tracking:
                    submit_frame(tracked.frame_index, tracked.rgb, tracked.alpha)
            finally:
                forward_tracking.close()
                packets.close()

            while pending:
                finish_oldest()
            output = writer.finalize()
        except Exception:
            writer.abort()
            raise
        finally:
            encoder.shutdown(wait=True, cancel_futures=True)
            shutil.rmtree(reverse_masks_dir, ignore_errors=True)

        return RotoscopeResult(
            job_id=resolved_job_id,
            video=video,
            output=output,
        )


def _reindex_packets(
    packets: Iterator[FramePacket],
    index_offset: int,
) -> Iterator[FramePacket]:
    try:
        for packet in packets:
            yield FramePacket(index=index_offset + packet.index, rgb=packet.rgb)
    finally:
        packets.close()


def _iter_reverse_clip_frames(
    video_path: str | Path,
    *,
    clip_start: int,
    seed_offset: int,
    chunk_size: int,
    decoder: Decoder,
) -> Iterator[FramePacket]:
    """Decode small forward chunks and yield them backward with bounded RAM."""
    cursor = seed_offset
    while cursor >= 0:
        chunk_start = max(0, cursor - chunk_size + 1)
        expected = cursor - chunk_start + 1
        source = iter_video_frames(
            video_path,
            max_frames=expected,
            decoder=decoder,
            start_frame=clip_start + chunk_start,
        )
        try:
            chunk = list(source)
        finally:
            source.close()
        if len(chunk) != expected:
            raise ValueError(
                f"Expected {expected} reverse frames, decoded {len(chunk)}"
            )
        for packet in reversed(chunk):
            yield FramePacket(index=chunk_start + packet.index, rgb=packet.rgb)
        cursor = chunk_start - 1


def _mask_filename(frame_index: int) -> str:
    return f"frame_{frame_index:06d}.png"


def _save_alpha_mask(alpha: np.ndarray, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.partial")
    try:
        Image.fromarray(np.asarray(alpha, dtype=np.uint8)).save(
            temporary,
            format="PNG",
            compress_level=1,
        )
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


__all__ = [
    "RotoscopeResult",
    "RotoscopeService",
    "SelectionPreview",
    "SelectionUpdate",
]
