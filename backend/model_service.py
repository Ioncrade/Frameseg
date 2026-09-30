from __future__ import annotations

import threading
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
import torch
from transformers import Sam2VideoModel, Sam2VideoProcessor

from media_loader import FramePacket


Point = tuple[float, float]
Box = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class ObjectPrompt:
    """A sparse cue used by SAM2 to produce a full-object mask."""

    object_id: int = 1
    points: tuple[Point, ...] = ()
    labels: tuple[int, ...] = ()
    box: Box | None = None

    def __post_init__(self) -> None:
        if self.object_id < 1:
            raise ValueError("object_id must be positive")
        if not self.points and self.box is None:
            raise ValueError("Provide at least one point or a bounding box")
        if len(self.points) != len(self.labels):
            raise ValueError("Every point must have a matching label")
        if any(label not in {0, 1} for label in self.labels):
            raise ValueError("Point labels must be 1 (foreground) or 0 (background)")
        if self.box is not None:
            x1, y1, x2, y2 = self.box
            if x2 <= x1 or y2 <= y1:
                raise ValueError("box must use (x1, y1, x2, y2) with positive area")

    @classmethod
    def positive_point(cls, x: float, y: float, object_id: int = 1) -> ObjectPrompt:
        return cls(object_id=object_id, points=((x, y),), labels=(1,))

    @classmethod
    def bounding_box(
        cls,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        object_id: int = 1,
    ) -> ObjectPrompt:
        return cls(object_id=object_id, box=(x1, y1, x2, y2))


@dataclass(frozen=True, slots=True)
class ObjectSelection:
    object_id: int
    mask_logits: np.ndarray
    object_score: float
    width: int
    height: int

    @property
    def binary_mask(self) -> np.ndarray:
        return self.mask_logits > 0.0


@dataclass(frozen=True, slots=True)
class SelectionSessionResult:
    session_id: str
    selection: ObjectSelection


@dataclass(frozen=True, slots=True)
class TrackedFrame:
    frame_index: int
    rgb: np.ndarray
    alpha: np.ndarray
    object_ids: tuple[int, ...]
    object_scores: tuple[float, ...]


@dataclass(slots=True)
class _SelectionState:
    inference_session: object
    object_id: int
    width: int
    height: int
    selection: ObjectSelection


class Sam2ModelService:
    """Long-lived, serialized SAM2 inference service for selection and tracking."""

    def __init__(
        self,
        model_id: str = "facebook/sam2.1-hiera-base-plus",
        device: str | torch.device | None = None,
        offload_state_to_cpu: bool = False,
        max_selection_sessions: int = 16,
    ) -> None:
        requested_device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if max_selection_sessions < 1:
            raise ValueError("max_selection_sessions must be positive")

        self.model_id = model_id
        self.device = requested_device
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.state_device = torch.device("cpu") if offload_state_to_cpu else self.device
        self.model: Sam2VideoModel | None = None
        self.processor: Sam2VideoProcessor | None = None
        self.max_selection_sessions = max_selection_sessions
        self._selection_sessions: dict[str, _SelectionState] = {}
        self._lock = threading.RLock()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def load(self) -> None:
        """Load the model once; subsequent calls are no-ops."""
        with self._lock:
            if self.is_loaded:
                return
            model = Sam2VideoModel.from_pretrained(self.model_id)
            model.to(self.device, dtype=self.dtype)
            model.eval()
            processor = Sam2VideoProcessor.from_pretrained(self.model_id)
            self.model = model
            self.processor = processor

    def unload(self) -> None:
        """Release this checkpoint and its sessions at an idle job boundary."""
        with self._lock:
            for state in self._selection_sessions.values():
                self._dispose_session(state.inference_session)
            self._selection_sessions.clear()
            model, self.model = self.model, None
            processor, self.processor = self.processor, None

        del model, processor
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def select_object(self, frame: np.ndarray, prompt: ObjectPrompt) -> ObjectSelection:
        """One-shot convenience wrapper for a non-refinable object selection."""
        result = self.start_selection(frame, prompt)
        try:
            return result.selection
        finally:
            self.close_selection(result.session_id)

    def start_selection(
        self,
        frame: np.ndarray,
        prompt: ObjectPrompt,
    ) -> SelectionSessionResult:
        """Start a refinable first-frame session and return the full-object mask."""
        self.load()
        rgb = _validate_rgb(frame)
        height, width = rgb.shape[:2]
        _validate_prompt_bounds(prompt, width, height)

        with self._lock, torch.inference_mode():
            if len(self._selection_sessions) >= self.max_selection_sessions:
                raise RuntimeError(
                    "Too many active selection sessions; close or accept an existing session"
                )
            model, processor = self._components()
            session = processor.init_video_session(
                video=None,
                inference_device=self.device,
                inference_state_device=self.state_device,
                video_storage_device=self.device,
                max_vision_features_cache_size=1,
                dtype=self.dtype,
            )
            self._add_prompt(session, prompt, (height, width))
            output = model(
                inference_session=session,
                frame=self._preprocess_frame(rgb),
                frame_idx=0,
            )
            selection = self._selection_from_output(
                output,
                object_id=prompt.object_id,
                original_size=(height, width),
            )
            session_id = uuid.uuid4().hex
            self._selection_sessions[session_id] = _SelectionState(
                inference_session=session,
                object_id=prompt.object_id,
                width=width,
                height=height,
                selection=selection,
            )

        return SelectionSessionResult(session_id=session_id, selection=selection)

    def refine_selection(
        self,
        session_id: str,
        additions: Iterable[Point] = (),
        subtractions: Iterable[Point] = (),
    ) -> ObjectSelection:
        """Update a full-object mask with positive and negative corrective clicks."""
        positive_points = tuple(additions)
        negative_points = tuple(subtractions)
        if not positive_points and not negative_points:
            raise ValueError("Provide at least one addition or subtraction point")

        with self._lock, torch.inference_mode():
            state = self._get_selection_state(session_id)
            points = positive_points + negative_points
            prompt = ObjectPrompt(
                object_id=state.object_id,
                points=points,
                labels=(1,) * len(positive_points) + (0,) * len(negative_points),
            )
            _validate_prompt_bounds(prompt, state.width, state.height)
            _model, processor = self._components()
            processor.add_inputs_to_inference_session(
                inference_session=state.inference_session,
                frame_idx=0,
                obj_ids=state.object_id,
                input_points=[[[list(point) for point in prompt.points]]],
                input_labels=[[list(prompt.labels)]],
                original_size=(state.height, state.width),
                clear_old_inputs=False,
            )
            model, _processor = self._components()
            output = model(inference_session=state.inference_session, frame_idx=0)
            state.selection = self._selection_from_output(
                output,
                object_id=state.object_id,
                original_size=(state.height, state.width),
            )
            return state.selection

    def get_selection(self, session_id: str) -> ObjectSelection:
        """Return the latest accepted/refined mask without copying model state."""
        with self._lock:
            return self._get_selection_state(session_id).selection

    def close_selection(self, session_id: str) -> None:
        """Release a selection session after acceptance or cancellation."""
        with self._lock:
            state = self._selection_sessions.pop(session_id, None)
            if state is not None:
                self._dispose_session(state.inference_session)

    def track_frames(
        self,
        frames: Iterable[FramePacket],
        initial_mask: np.ndarray,
        object_id: int = 1,
        reverse: bool = False,
    ) -> Iterator[TrackedFrame]:
        """Track an accepted mask forward or backward over an ordered frame stream."""
        self.load()
        frame_iterator = iter(frames)
        try:
            first_packet = next(frame_iterator)
        except StopIteration as error:
            raise ValueError("frames cannot be empty") from error

        first_rgb = _validate_rgb(first_packet.rgb)
        height, width = first_rgb.shape[:2]
        accepted_mask = np.asarray(initial_mask)
        if accepted_mask.shape != (height, width):
            raise ValueError(
                f"Initial mask shape {accepted_mask.shape} does not match frame shape {(height, width)}"
            )

        with self._lock, torch.inference_mode():
            model, processor = self._components()
            session = processor.init_video_session(
                video=None,
                inference_device=self.device,
                inference_state_device=self.state_device,
                video_storage_device=self.device,
                max_vision_features_cache_size=1,
                dtype=self.dtype,
            )
            session.video_height = height
            session.video_width = width
            processor.add_inputs_to_inference_session(
                inference_session=session,
                frame_idx=first_packet.index,
                obj_ids=object_id,
                input_masks=accepted_mask,
            )

            try:
                yield self._track_one(session, first_packet, reverse=reverse)
                for packet in frame_iterator:
                    if packet.rgb.shape[:2] != (height, width):
                        raise ValueError(
                            f"Frame {packet.index} changed resolution from "
                            f"{(width, height)} to {(packet.rgb.shape[1], packet.rgb.shape[0])}"
                        )
                    yield self._track_one(session, packet, reverse=reverse)
            finally:
                self._dispose_session(session)

    def _track_one(
        self,
        session,
        packet: FramePacket,
        *,
        reverse: bool,
    ) -> TrackedFrame:
        model, _processor = self._components()
        output = model(
            inference_session=session,
            frame=self._preprocess_frame(packet.rgb),
            frame_idx=packet.index,
            reverse=reverse,
        )
        mask_tensor = self._post_process_mask_tensor(
            output.pred_masks,
            (session.video_height, session.video_width),
        )
        alpha = self._mask_tensor_to_alpha(mask_tensor)
        scores = tuple(
            float(value)
            for value in output.object_score_logits.detach().float().cpu().reshape(-1)
        )

        # The model has already encoded this frame into temporal memory.
        session.processed_frames.pop(packet.index, None)
        self._prune_tracking_history(session, packet.index, reverse=reverse)

        return TrackedFrame(
            frame_index=packet.index,
            rgb=packet.rgb,
            alpha=alpha,
            object_ids=tuple(output.object_ids),
            object_scores=scores,
        )

    def _add_prompt(self, session, prompt: ObjectPrompt, original_size: tuple[int, int]) -> None:
        _model, processor = self._components()
        input_points = None
        input_labels = None
        input_boxes = None
        if prompt.points:
            input_points = [[[list(point) for point in prompt.points]]]
            input_labels = [[list(prompt.labels)]]
        if prompt.box is not None:
            input_boxes = [[list(prompt.box)]]

        processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=0,
            obj_ids=prompt.object_id,
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            original_size=original_size,
        )

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        _model, processor = self._components()
        processed = processor.video_processor(
            videos=frame,
            device=self.device,
            return_tensors="pt",
        )
        return processed.pixel_values_videos[0, 0].to(dtype=self.dtype)

    def _post_process_masks(
        self,
        pred_masks: torch.Tensor,
        original_size: tuple[int, int],
    ) -> np.ndarray:
        return (
            self._post_process_mask_tensor(pred_masks, original_size)
            .detach()
            .float()
            .cpu()
            .numpy()
        )

    def _post_process_mask_tensor(
        self,
        pred_masks: torch.Tensor,
        original_size: tuple[int, int],
    ) -> torch.Tensor:
        _model, processor = self._components()
        return processor.post_process_masks(
            [pred_masks],
            original_sizes=[[original_size[0], original_size[1]]],
            binarize=False,
        )[0]

    @staticmethod
    def _mask_tensor_to_alpha(mask_logits: torch.Tensor) -> np.ndarray:
        """Union masks and transfer only the final 8-bit alpha plane to CPU."""
        if mask_logits.ndim < 2:
            raise ValueError("mask_logits must have at least two dimensions")
        height, width = mask_logits.shape[-2:]
        probabilities = torch.sigmoid(mask_logits.float()).reshape(-1, height, width)
        union = 1.0 - torch.prod(1.0 - probabilities, dim=0)
        return (
            union.clamp_(0, 1)
            .mul_(255)
            .round_()
            .to(dtype=torch.uint8)
            .cpu()
            .numpy()
        )

    def _selection_from_output(
        self,
        output,
        object_id: int,
        original_size: tuple[int, int],
    ) -> ObjectSelection:
        height, width = original_size
        all_masks = self._post_process_masks(output.pred_masks, original_size)
        mask_logits = all_masks.reshape(-1, height, width)[0]
        score = float(output.object_score_logits.detach().float().cpu().reshape(-1)[0])
        return ObjectSelection(
            object_id=object_id,
            mask_logits=mask_logits,
            object_score=score,
            width=width,
            height=height,
        )

    def _prune_tracking_history(
        self,
        session,
        current_frame: int,
        *,
        reverse: bool,
    ) -> None:
        model, _processor = self._components()
        history_size = max(
            model.num_maskmem - 1,
            model.config.max_object_pointers_in_encoder - 1,
        )
        history_boundary = (
            current_frame + history_size if reverse else current_frame - history_size
        )

        for object_index in range(session.get_obj_num()):
            non_conditioning = session.output_dict_per_obj[object_index][
                "non_cond_frame_outputs"
            ]
            for frame_index in tuple(non_conditioning):
                if (
                    frame_index > history_boundary
                    if reverse
                    else frame_index < history_boundary
                ):
                    non_conditioning.pop(frame_index, None)

            tracked = session.frames_tracked_per_obj[object_index]
            for frame_index in tuple(tracked):
                if (
                    frame_index > history_boundary
                    if reverse
                    else frame_index < history_boundary
                ):
                    tracked.pop(frame_index, None)

    def _components(self) -> tuple[Sam2VideoModel, Sam2VideoProcessor]:
        if self.model is None or self.processor is None:
            raise RuntimeError("SAM2 is not loaded")
        return self.model, self.processor

    def _get_selection_state(self, session_id: str) -> _SelectionState:
        try:
            return self._selection_sessions[session_id]
        except KeyError as error:
            raise KeyError(f"Unknown or expired selection session: {session_id}") from error

    @staticmethod
    def _dispose_session(session: object) -> None:
        """Deterministically release frame, prompt, feature, and history tensors."""
        reset = getattr(session, "reset_inference_session", None)
        if reset is not None:
            reset()
        processed_frames = getattr(session, "processed_frames", None)
        if processed_frames is not None:
            processed_frames.clear()


def _validate_rgb(frame: np.ndarray) -> np.ndarray:
    rgb = np.asarray(frame, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("frame must have shape (height, width, 3)")
    return rgb


def _validate_prompt_bounds(prompt: ObjectPrompt, width: int, height: int) -> None:
    for x, y in prompt.points:
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"Prompt point {(x, y)} is outside the {width}x{height} frame")
    if prompt.box is not None:
        x1, y1, x2, y2 = prompt.box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Prompt box {prompt.box} is outside the {width}x{height} frame")


__all__ = [
    "Box",
    "ObjectPrompt",
    "ObjectSelection",
    "Point",
    "Sam2ModelService",
    "SelectionSessionResult",
    "TrackedFrame",
]
