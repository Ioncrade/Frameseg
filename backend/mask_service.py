from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

import av
import numpy as np
import torch
from PIL import Image


OutputCodec = Literal["vp9_alpha", "qtrle", "prores_4444"]


@dataclass(frozen=True, slots=True)
class FrameArtifacts:
    frame_index: int
    rgb_path: Path | None = None
    alpha_path: Path | None = None
    rgba_path: Path | None = None


@dataclass(frozen=True, slots=True)
class OutputManifest:
    output_dir: Path
    frame_count: int
    fps: float
    video_path: Path


class MaskOutputService:
    """Incrementally encode an alpha video with optional debug frame artifacts."""

    def __init__(
        self,
        output_dir: str | Path,
        fps: float,
        filename: str | None = None,
        keep_frame_artifacts: bool = False,
        transparent_alpha_threshold: int = 0,
        output_codec: OutputCodec = "prores_4444",
        encoder_threads: int = 0,
        vp9_crf: int = 18,
        vp9_cpu_used: int = 5,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        if not 0 <= transparent_alpha_threshold <= 255:
            raise ValueError("transparent_alpha_threshold must be between 0 and 255")
        if output_codec not in {"vp9_alpha", "qtrle", "prores_4444"}:
            raise ValueError(
                "output_codec must be 'vp9_alpha', 'qtrle', or 'prores_4444'"
            )
        if isinstance(encoder_threads, bool) or encoder_threads < 0:
            raise ValueError("encoder_threads cannot be negative")
        if isinstance(vp9_crf, bool) or not 0 <= vp9_crf <= 63:
            raise ValueError("vp9_crf must be between 0 and 63")
        if isinstance(vp9_cpu_used, bool) or not 0 <= vp9_cpu_used <= 8:
            raise ValueError("vp9_cpu_used must be between 0 and 8")

        suffix = ".webm" if output_codec == "vp9_alpha" else ".mov"
        resolved_filename = filename or f"rotoscope_alpha{suffix}"
        if (
            Path(resolved_filename).name != resolved_filename
            or Path(resolved_filename).suffix.lower() != suffix
        ):
            raise ValueError(f"filename must be a plain {suffix} filename")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.keep_frame_artifacts = keep_frame_artifacts
        self.transparent_alpha_threshold = transparent_alpha_threshold
        self.output_codec = output_codec
        self.encoder_threads = encoder_threads
        self.vp9_crf = vp9_crf
        self.vp9_cpu_used = vp9_cpu_used
        self.video_path = self.output_dir / resolved_filename
        self.partial_path = self.video_path.with_name(
            f".{self.video_path.stem}.partial{self.video_path.suffix}"
        )
        self.rgb_dir = self.output_dir / "rgb_frames"
        self.alpha_dir = self.output_dir / "alpha_masks"
        self.rgba_dir = self.output_dir / "rgba_frames"
        self._container: av.container.OutputContainer | None = None
        self._stream = None
        self._frame_count = 0
        self._closed = False

        if self.keep_frame_artifacts:
            for directory in (self.rgb_dir, self.alpha_dir, self.rgba_dir):
                directory.mkdir(parents=True, exist_ok=True)

    def write_frame(
        self,
        frame_index: int,
        rgb: np.ndarray,
        mask_logits: torch.Tensor | np.ndarray | None = None,
        *,
        alpha: np.ndarray | None = None,
    ) -> FrameArtifacts:
        """Encode one frame immediately and retain no prior frame arrays."""
        if self._closed:
            raise RuntimeError("The output writer is already closed")
        if frame_index != self._frame_count:
            raise ValueError(
                f"Expected sequential frame {self._frame_count}, received {frame_index}"
            )

        rgb_array = np.asarray(rgb, dtype=np.uint8)
        if rgb_array.ndim != 3 or rgb_array.shape[-1] != 3:
            raise ValueError("rgb must have shape (height, width, 3)")

        if alpha is None:
            if mask_logits is None:
                raise ValueError("Provide alpha or mask_logits")
            alpha_array = mask_logits_to_alpha(mask_logits)
        else:
            alpha_array = np.asarray(alpha, dtype=np.uint8)
        if alpha_array.shape != rgb_array.shape[:2]:
            raise ValueError(
                f"Mask size {alpha_array.shape} does not match RGB size {rgb_array.shape[:2]}"
            )

        # Straight alpha has undefined RGB where alpha is zero. Clearing those
        # invisible pixels prevents the codec from storing the entire hidden source.
        transparent = alpha_array <= self.transparent_alpha_threshold
        if self.output_codec == "qtrle":
            # QTRLE's native format is ARGB. Supplying it directly avoids an FFmpeg
            # RGBA -> ARGB conversion and an additional full-frame allocation.
            encoded_pixels = np.empty((*rgb_array.shape[:2], 4), dtype=np.uint8)
            encoded_pixels[..., 0] = alpha_array
            encoded_pixels[..., 1:] = rgb_array
            encoded_pixels[..., 1:][transparent] = 0
            input_format = "argb"
            rgba = None
        else:
            encoded_pixels = np.empty((*rgb_array.shape[:2], 4), dtype=np.uint8)
            encoded_pixels[..., :3] = rgb_array
            encoded_pixels[..., :3][transparent] = 0
            encoded_pixels[..., 3] = alpha_array
            input_format = "rgba"
            rgba = encoded_pixels

        artifacts = self._write_debug_artifacts(
            frame_index, rgb_array, alpha_array, rgba
        )
        try:
            if self._container is None:
                self._open_encoder(width=rgb_array.shape[1], height=rgb_array.shape[0])
            assert self._container is not None and self._stream is not None
            video_frame = av.VideoFrame.from_ndarray(encoded_pixels, format=input_format)
            for packet in self._stream.encode(video_frame):
                self._container.mux(packet)
            self._frame_count += 1
            return artifacts
        except Exception:
            self.abort()
            raise

    def finalize(self, filename: str | None = None) -> OutputManifest:
        """Flush the streaming alpha encoder and atomically publish the video."""
        if filename is not None and filename != self.video_path.name:
            raise ValueError("The output filename must be chosen before encoding starts")
        if self._frame_count == 0 or self._container is None or self._stream is None:
            raise ValueError("Cannot finalize a video without any written frames")
        if self._closed:
            raise RuntimeError("The output writer is already closed")

        try:
            for packet in self._stream.encode():
                self._container.mux(packet)
            self._container.close()
            self._closed = True
            self._container = None
            self.partial_path.replace(self.video_path)
        except Exception:
            self.abort()
            raise

        return OutputManifest(
            output_dir=self.output_dir,
            frame_count=self._frame_count,
            fps=self.fps,
            video_path=self.video_path,
        )

    def abort(self) -> None:
        """Close the encoder and discard an incomplete output."""
        container, self._container = self._container, None
        self._closed = True
        if container is not None:
            try:
                container.close()
            except Exception:
                pass
        self.partial_path.unlink(missing_ok=True)

    def _open_encoder(self, width: int, height: int) -> None:
        self.partial_path.unlink(missing_ok=True)
        container_format = "webm" if self.output_codec == "vp9_alpha" else "mov"
        container = av.open(
            str(self.partial_path),
            mode="w",
            format=container_format,
        )
        try:
            codec_name = {
                "vp9_alpha": "libvpx-vp9",
                "qtrle": "qtrle",
                "prores_4444": "prores_ks",
            }[self.output_codec]
            stream = container.add_stream(
                codec_name, rate=Fraction(str(self.fps)).limit_denominator(100_000)
            )
            stream.width = width
            stream.height = height
            if self.output_codec == "vp9_alpha":
                stream.pix_fmt = "yuva420p"
                stream.bit_rate = 0
                stream.options = {
                    "crf": str(self.vp9_crf),
                    "deadline": "good",
                    "cpu-used": str(self.vp9_cpu_used),
                    "row-mt": "1",
                    "tile-columns": "2" if width >= 1920 else "1",
                    "frame-parallel": "1",
                    "auto-alt-ref": "0",
                }
                stream.metadata["alpha_mode"] = "1"
            elif self.output_codec == "qtrle":
                stream.pix_fmt = "argb"
            else:
                stream.pix_fmt = "yuva444p10le"
                stream.options = {
                    "profile": "4",
                    "alpha_bits": "8",
                }
            if self.encoder_threads:
                stream.codec_context.thread_count = self.encoder_threads
        except Exception:
            container.close()
            raise
        self._container = container
        self._stream = stream

    def _write_debug_artifacts(
        self,
        frame_index: int,
        rgb: np.ndarray,
        alpha: np.ndarray,
        rgba: np.ndarray | None,
    ) -> FrameArtifacts:
        if not self.keep_frame_artifacts:
            return FrameArtifacts(frame_index=frame_index)

        if rgba is None:
            rgba = np.empty((*rgb.shape[:2], 4), dtype=np.uint8)
            rgba[..., :3] = rgb
            rgba[..., :3][alpha <= self.transparent_alpha_threshold] = 0
            rgba[..., 3] = alpha

        filename = f"frame_{frame_index:06d}.png"
        artifacts = FrameArtifacts(
            frame_index=frame_index,
            rgb_path=self.rgb_dir / filename,
            alpha_path=self.alpha_dir / filename,
            rgba_path=self.rgba_dir / filename,
        )
        assert artifacts.rgb_path is not None
        assert artifacts.alpha_path is not None
        assert artifacts.rgba_path is not None
        _save_png_atomic(Image.fromarray(rgb), artifacts.rgb_path)
        _save_png_atomic(Image.fromarray(alpha), artifacts.alpha_path)
        _save_png_atomic(Image.fromarray(rgba), artifacts.rgba_path)
        return artifacts


def mask_logits_to_alpha(mask_logits: torch.Tensor | np.ndarray) -> np.ndarray:
    """Union full-object mask logits into one soft 8-bit alpha channel."""
    if isinstance(mask_logits, torch.Tensor):
        logits = mask_logits.detach().float()
    else:
        logits = torch.as_tensor(mask_logits, dtype=torch.float32)

    if logits.ndim < 2:
        raise ValueError("mask_logits must have at least two dimensions")
    height, width = logits.shape[-2:]
    probabilities = torch.sigmoid(logits).reshape(-1, height, width)
    union_probability = 1.0 - torch.prod(1.0 - probabilities, dim=0)
    return (
        union_probability.clamp(0, 1)
        .mul(255)
        .round()
        .to(dtype=torch.uint8)
        .cpu()
        .numpy()
    )


def _save_png_atomic(image: Image.Image, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.partial")
    try:
        image.save(temporary, format="PNG")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


__all__ = [
    "FrameArtifacts",
    "MaskOutputService",
    "OutputCodec",
    "OutputManifest",
    "mask_logits_to_alpha",
]
