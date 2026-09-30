"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
  type MouseEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";

import {
  API_BASE,
  SAM2_MODEL_OPTIONS,
  assetUrl,
  cancelSelection,
  createSelection,
  deleteVideo,
  getRenderJob,
  refineSelection,
  startRender,
  uploadVideo,
  type RenderJob,
  type Sam2ModelVariant,
  type Selection,
  type VideoUpload,
} from "@/lib/api";

type EditMode = "add" | "subtract";
type TimelineDragMode = "move" | "resize-start" | "resize-end" | "seed";

type TimelineDragState = {
  mode: TimelineDragMode;
  pointerId: number;
  pointerX: number;
  startFrame: number;
  frameCount: number;
  seedFrame: number;
  currentSeedFrame: number;
};

const MIN_ZOOM = 1;
const MAX_ZOOM = 8;

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, value));
}

function isEditableTarget(target: EventTarget | null) {
  if (!(target instanceof HTMLElement)) return false;
  return (
    target.isContentEditable ||
    ["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(target.tagName)
  );
}

export default function RotoscopeStudio() {
  const inputRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const timelineRef = useRef<HTMLDivElement>(null);
  const timelineDragRef = useRef<TimelineDragState | null>(null);
  const panOriginRef = useRef<{
    pointerX: number;
    pointerY: number;
    panX: number;
    panY: number;
  } | null>(null);
  const didPanRef = useRef(false);
  const suppressClickUntilRef = useRef(0);
  const [video, setVideo] = useState<VideoUpload | null>(null);
  const [selection, setSelection] = useState<Selection | null>(null);
  const [job, setJob] = useState<RenderJob | null>(null);
  const [fileName, setFileName] = useState("");
  const [mode, setMode] = useState<EditMode>("add");
  const [modelVariant, setModelVariant] =
    useState<Sam2ModelVariant>("base_plus");
  const [clipStart, setClipStart] = useState(0);
  const [clipFrames, setClipFrames] = useState(30);
  const [previewFrame, setPreviewFrame] = useState(0);
  const [timelinePreviewFrame, setTimelinePreviewFrame] = useState<number | null>(
    null,
  );
  const [timelineDragMode, setTimelineDragMode] =
    useState<TimelineDragMode | null>(null);
  const [zoom, setZoom] = useState(MIN_ZOOM);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [spacePressed, setSpacePressed] = useState(false);
  const [isPanning, setIsPanning] = useState(false);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const frameUrl = video
    ? `${assetUrl(video.first_frame_url)}?frame_index=${previewFrame}`
    : null;
  const alphaUrl = selection ? assetUrl(selection.alpha_url) : null;
  const renderLocked = job !== null;

  const resetViewport = useCallback(() => {
    panOriginRef.current = null;
    didPanRef.current = false;
    suppressClickUntilRef.current = 0;
    setZoom(MIN_ZOOM);
    setPan({ x: 0, y: 0 });
    setSpacePressed(false);
    setIsPanning(false);
  }, []);

  const constrainPan = useCallback(
    (candidate: { x: number; y: number }, zoomLevel: number) => {
      const stage = stageRef.current;
      const canvas = canvasRef.current;
      if (!stage || !canvas || zoomLevel <= MIN_ZOOM) {
        return { x: 0, y: 0 };
      }

      const stageStyle = window.getComputedStyle(stage);
      const viewportWidth =
        stage.clientWidth -
        Number.parseFloat(stageStyle.paddingLeft) -
        Number.parseFloat(stageStyle.paddingRight);
      const viewportHeight =
        stage.clientHeight -
        Number.parseFloat(stageStyle.paddingTop) -
        Number.parseFloat(stageStyle.paddingBottom);
      const maxX = Math.max(
        0,
        (canvas.offsetWidth * zoomLevel - viewportWidth) / 2,
      );
      const maxY = Math.max(
        0,
        (canvas.offsetHeight * zoomLevel - viewportHeight) / 2,
      );
      return {
        x: clamp(candidate.x, -maxX, maxX),
        y: clamp(candidate.y, -maxY, maxY),
      };
    },
    [],
  );

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (
        event.code !== "Space" ||
        event.repeat ||
        isEditableTarget(event.target) ||
        !video ||
        busy ||
        renderLocked
      ) {
        return;
      }
      event.preventDefault();
      setSpacePressed(true);
    };

    const finishSpaceGesture = (event?: KeyboardEvent | Event) => {
      if (event instanceof KeyboardEvent && event.code !== "Space") return;
      if (event instanceof KeyboardEvent) event.preventDefault();
      if (didPanRef.current) {
        suppressClickUntilRef.current = performance.now() + 250;
      }
      didPanRef.current = false;
      panOriginRef.current = null;
      setSpacePressed(false);
      setIsPanning(false);
    };

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", finishSpaceGesture);
    window.addEventListener("blur", finishSpaceGesture);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", finishSpaceGesture);
      window.removeEventListener("blur", finishSpaceGesture);
    };
  }, [busy, renderLocked, video]);

  const onStageWheel = useCallback((event: WheelEvent) => {
    if (!video || busy || renderLocked) return;

    event.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;

    const nextZoom = clamp(
      zoom * Math.exp(-event.deltaY * 0.0015),
      MIN_ZOOM,
      MAX_ZOOM,
    );
    if (Math.abs(nextZoom - zoom) < 0.001) return;

    const bounds = canvas.getBoundingClientRect();
    const pointerFromCenter = {
      x: event.clientX - (bounds.left + bounds.right) / 2,
      y: event.clientY - (bounds.top + bounds.bottom) / 2,
    };
    const scaleRatio = nextZoom / zoom;
    const nextPan = constrainPan(
      {
        x: pan.x + pointerFromCenter.x * (1 - scaleRatio),
        y: pan.y + pointerFromCenter.y * (1 - scaleRatio),
      },
      nextZoom,
    );

    setZoom(nextZoom);
    setPan(nextPan);
  }, [busy, constrainPan, pan, renderLocked, video, zoom]);

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;
    stage.addEventListener("wheel", onStageWheel, { passive: false });
    return () => stage.removeEventListener("wheel", onStageWheel);
  }, [onStageWheel]);

  function onStagePointerDown(event: ReactPointerEvent<HTMLDivElement>) {
    if (
      !spacePressed ||
      zoom <= MIN_ZOOM ||
      busy ||
      renderLocked ||
      event.button !== 0
    ) {
      return;
    }

    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    panOriginRef.current = {
      pointerX: event.clientX,
      pointerY: event.clientY,
      panX: pan.x,
      panY: pan.y,
    };
    didPanRef.current = false;
    setIsPanning(true);
  }

  function onStagePointerMove(event: ReactPointerEvent<HTMLDivElement>) {
    const origin = panOriginRef.current;
    if (!origin) return;

    event.preventDefault();
    const deltaX = event.clientX - origin.pointerX;
    const deltaY = event.clientY - origin.pointerY;
    if (Math.hypot(deltaX, deltaY) > 2) didPanRef.current = true;
    setPan(
      constrainPan(
        { x: origin.panX + deltaX, y: origin.panY + deltaY },
        zoom,
      ),
    );
  }

  function finishPanning(event: ReactPointerEvent<HTMLDivElement>) {
    if (!panOriginRef.current) return;
    if (didPanRef.current) {
      suppressClickUntilRef.current = performance.now() + 250;
    }
    didPanRef.current = false;
    panOriginRef.current = null;
    setIsPanning(false);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  useEffect(() => {
    if (!frameUrl || !canvasRef.current || !video) return;
    let cancelled = false;
    const canvas = canvasRef.current;

    async function draw() {
      const frame = await loadImage(frameUrl!);
      const alpha = alphaUrl ? await loadImage(alphaUrl) : null;
      if (cancelled) return;

      canvas.width = video!.metadata.width;
      canvas.height = video!.metadata.height;
      const context = canvas.getContext("2d", { willReadFrequently: false });
      if (!context) return;
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.drawImage(frame, 0, 0, canvas.width, canvas.height);

      if (alpha) {
        const overlay = document.createElement("canvas");
        overlay.width = canvas.width;
        overlay.height = canvas.height;
        const overlayContext = overlay.getContext("2d", {
          willReadFrequently: true,
        });
        if (!overlayContext) return;
        overlayContext.drawImage(alpha, 0, 0, canvas.width, canvas.height);
        const pixels = overlayContext.getImageData(
          0,
          0,
          canvas.width,
          canvas.height,
        );
        for (let index = 0; index < pixels.data.length; index += 4) {
          const coverage = pixels.data[index] / 255;
          pixels.data[index] = 80;
          pixels.data[index + 1] = 244;
          pixels.data[index + 2] = 190;
          pixels.data[index + 3] = Math.round(coverage * 178);
        }
        overlayContext.putImageData(pixels, 0, 0);
        context.drawImage(overlay, 0, 0);
      }
    }

    draw().catch((drawError: unknown) => {
      if (!cancelled) {
        setError(
          drawError instanceof Error ? drawError.message : "Preview failed",
        );
      }
    });
    return () => {
      cancelled = true;
    };
  }, [alphaUrl, frameUrl, video]);

  useEffect(() => {
    if (!job || !["queued", "running"].includes(job.status)) return;
    let cancelled = false;
    let timeout: ReturnType<typeof setTimeout>;

    async function poll() {
      try {
        const nextJob = await getRenderJob(job!.job_id);
        if (cancelled) return;
        setJob(nextJob);
        if (["queued", "running"].includes(nextJob.status)) {
          timeout = setTimeout(poll, 1500);
        }
      } catch (pollError) {
        if (!cancelled) {
          setError(
            pollError instanceof Error ? pollError.message : "Status check failed",
          );
        }
      }
    }

    timeout = setTimeout(poll, 800);
    return () => {
      cancelled = true;
      clearTimeout(timeout);
    };
  }, [job]);

  const handleFile = useCallback(
    async (file: File) => {
      if (!file.name.toLowerCase().match(/\.(mp4|mov)$/)) {
        setError("Choose an H.264 or H.265 video in MP4 or MOV format.");
        return;
      }
      if (selection && !renderLocked) {
        void cancelSelection(selection.selection_id).catch(() => undefined);
      }
      setBusy(true);
      setError(null);
      resetViewport();
      timelineDragRef.current = null;
      setTimelinePreviewFrame(null);
      setTimelineDragMode(null);
      setVideo(null);
      setSelection(null);
      setJob(null);
      setFileName(file.name);
      try {
        const uploaded = await uploadVideo(file);
        const initialFrameCount = Math.min(
          uploaded.metadata.total_frames,
          uploaded.metadata.max_frames,
        );
        setClipStart(0);
        setPreviewFrame(0);
        setClipFrames(initialFrameCount);
        setVideo(uploaded);
      } catch (uploadError) {
        setError(
          uploadError instanceof Error ? uploadError.message : "Upload failed",
        );
      } finally {
        setBusy(false);
      }
    },
    [renderLocked, resetViewport, selection],
  );

  async function onCanvasClick(event: MouseEvent<HTMLCanvasElement>) {
    if (performance.now() < suppressClickUntilRef.current) {
      suppressClickUntilRef.current = 0;
      return;
    }
    if (spacePressed || isPanning) return;
    if (!video || busy || renderLocked) return;
    if (previewFrame < clipStart || previewFrame > clipEnd) {
      setError("Choose a selection frame inside the clip range.");
      return;
    }
    const canvas = event.currentTarget;
    const bounds = canvas.getBoundingClientRect();
    const point = {
      x: ((event.clientX - bounds.left) / bounds.width) * canvas.width,
      y: ((event.clientY - bounds.top) / bounds.height) * canvas.height,
    };

    setBusy(true);
    setError(null);
    try {
      if (!selection) {
        setSelection(
          await createSelection(
            video.video_id,
            point,
            clipStart,
            clipFrames,
            previewFrame,
          ),
        );
        setMode("add");
      } else {
        setSelection(
          await refineSelection(selection.selection_id, point, mode),
        );
      }
    } catch (selectionError) {
      setError(
        selectionError instanceof Error
          ? selectionError.message
          : "Selection failed",
      );
    } finally {
      setBusy(false);
    }
  }

  async function renderVideo() {
    if (!video || !selection || busy) return;
    setBusy(true);
    setError(null);
    try {
      setJob(await startRender(selection.selection_id, modelVariant));
    } catch (renderError) {
      setError(
        renderError instanceof Error ? renderError.message : "Render failed",
      );
    } finally {
      setBusy(false);
    }
  }

  function clearLocalProject() {
    const canvas = canvasRef.current;
    if (canvas) {
      canvas.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
      canvas.width = 0;
      canvas.height = 0;
    }
    setVideo(null);
    setSelection(null);
    setJob(null);
    setFileName("");
    setError(null);
    setMode("add");
    setModelVariant("base_plus");
    setClipStart(0);
    setClipFrames(30);
    setPreviewFrame(0);
    resetViewport();
    timelineDragRef.current = null;
    setTimelinePreviewFrame(null);
    setTimelineDragMode(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  async function discardCurrentVideo() {
    if (!video || busy) return;
    const completedRenderWarning =
      job?.status === "completed"
        ? " Download the current render first if you need to keep it."
        : "";
    const confirmed = window.confirm(
      `Start a new video? The current upload, mask state, cached frames, and any rendered output will be permanently wiped from FrameSeg.${completedRenderWarning}`,
    );
    if (!confirmed) return;

    setBusy(true);
    setError(null);
    try {
      await deleteVideo(video.video_id);
      clearLocalProject();
    } catch (discardError) {
      setError(
        discardError instanceof Error
          ? discardError.message
          : "Could not clear the previous video",
      );
    } finally {
      setBusy(false);
    }
  }

  async function editClipRange() {
    if (!selection || job || busy) return;
    setBusy(true);
    setError(null);
    try {
      await cancelSelection(selection.selection_id);
      setSelection(null);
      setMode("add");
    } catch (rangeError) {
      setError(
        rangeError instanceof Error
          ? rangeError.message
          : "Could not unlock the clip range",
      );
    } finally {
      setBusy(false);
    }
  }

  const step = !video ? 1 : !selection ? 2 : !job ? 3 : 4;
  const progress = job ? Math.round(job.progress * 100) : 0;
  const selectedModel =
    SAM2_MODEL_OPTIONS.find((model) => model.variant === modelVariant) ??
    SAM2_MODEL_OPTIONS[2];
  const jobModel = job
    ? SAM2_MODEL_OPTIONS.find((model) => model.variant === job.model_variant)
    : null;
  const totalFrames = video?.metadata.total_frames ?? 0;
  const maxClipFrames = video
    ? Math.min(video.metadata.max_frames, totalFrames)
    : 30;
  const minClipFrames = video
    ? Math.min(video.metadata.min_clip_frames, maxClipFrames)
    : 30;
  const clipEnd = clipStart + clipFrames - 1;
  const selectionLeft = totalFrames ? (clipStart / totalFrames) * 100 : 0;
  const selectionWidth = totalFrames ? (clipFrames / totalFrames) * 100 : 100;
  const timelineFrames = video
    ? Array.from({ length: 8 }, (_, index) =>
        Math.round(((totalFrames - 1) * index) / 7),
      )
    : [];
  const displayedSeedFrame = timelinePreviewFrame ?? previewFrame;

  function beginTimelineDrag(
    event: ReactPointerEvent<HTMLElement>,
    dragMode: TimelineDragMode,
  ) {
    if (!video || busy || selection || event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    timelineRef.current?.setPointerCapture(event.pointerId);
    timelineDragRef.current = {
      mode: dragMode,
      pointerId: event.pointerId,
      pointerX: event.clientX,
      startFrame: clipStart,
      frameCount: clipFrames,
      seedFrame: previewFrame,
      currentSeedFrame: previewFrame,
    };
    setTimelinePreviewFrame(previewFrame);
    setTimelineDragMode(dragMode);
  }

  function onTimelinePointerMove(event: ReactPointerEvent<HTMLDivElement>) {
    const drag = timelineDragRef.current;
    if (!drag || drag.pointerId !== event.pointerId || totalFrames < 1) return;
    event.preventDefault();

  const bounds = event.currentTarget.getBoundingClientRect();
  if (bounds.width <= 0) return;
  const frameDelta = Math.round(
      ((event.clientX - drag.pointerX) / bounds.width) * totalFrames,
    );
    const originalEnd = drag.startFrame + drag.frameCount;
    let nextStart = drag.startFrame;
    let nextFrameCount = drag.frameCount;
    let nextSeed = drag.seedFrame;

    if (drag.mode === "move") {
      nextStart = clamp(
        drag.startFrame + frameDelta,
        0,
        totalFrames - drag.frameCount,
      );
      nextSeed = nextStart + (drag.seedFrame - drag.startFrame);
    } else if (drag.mode === "resize-start") {
      nextStart = clamp(
        drag.startFrame + frameDelta,
        Math.max(0, originalEnd - maxClipFrames),
        originalEnd - minClipFrames,
      );
      nextFrameCount = originalEnd - nextStart;
      nextSeed = clamp(drag.seedFrame, nextStart, originalEnd - 1);
    } else if (drag.mode === "resize-end") {
      const nextEnd = clamp(
        originalEnd + frameDelta,
        drag.startFrame + minClipFrames,
        Math.min(totalFrames, drag.startFrame + maxClipFrames),
      );
      nextFrameCount = nextEnd - drag.startFrame;
      nextSeed = clamp(
        drag.seedFrame,
        drag.startFrame,
        nextEnd - 1,
      );
    } else {
      nextSeed = clamp(
        drag.seedFrame + frameDelta,
        drag.startFrame,
        originalEnd - 1,
      );
    }

    if (drag.mode !== "seed") {
      setClipStart(nextStart);
      setClipFrames(nextFrameCount);
    }
    drag.currentSeedFrame = nextSeed;
    setTimelinePreviewFrame(nextSeed);
  }

  function finishTimelineDrag(event: ReactPointerEvent<HTMLDivElement>) {
    const drag = timelineDragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    event.preventDefault();
    timelineDragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    resetViewport();
    setPreviewFrame(drag.currentSeedFrame);
    setTimelinePreviewFrame(null);
    setTimelineDragMode(null);
  }

  return (
    <div className="studio-shell">
      <aside className="control-panel">
        <div>
          <p className="eyebrow">Workflow</p>
          <h2 className="panel-title">Create a clean cutout</h2>
        </div>

        <ol className="step-list">
          {[
            [1, "Import", "MP4 or MOV"],
            [2, "Range & select", "Choose frames"],
            [3, "Refine", "Add or subtract"],
            [4, "Render", "Alpha MOV"],
          ].map(([number, label, caption]) => (
            <li
              className={`step ${step === number ? "step-active" : ""} ${step > Number(number) ? "step-done" : ""}`}
              key={label}
            >
              <span className="step-number">{step > Number(number) ? "✓" : number}</span>
              <span>
                <strong>{label}</strong>
                <small>{caption}</small>
              </span>
            </li>
          ))}
        </ol>

        {!video ? (
          <label
            className={`drop-zone ${dragging ? "drop-zone-active" : ""}`}
            onDragEnter={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={() => setDragging(false)}
            onDrop={(event: DragEvent<HTMLLabelElement>) => {
              event.preventDefault();
              setDragging(false);
              const file = event.dataTransfer.files[0];
              if (file) void handleFile(file);
            }}
          >
            <UploadIcon />
            <strong>{busy ? "Preparing footage…" : "Drop footage here"}</strong>
            <span>or choose a file · up to 1,200 frames</span>
            <input
              ref={inputRef}
              type="file"
              accept="video/mp4,video/quicktime,.mp4,.mov"
              disabled={busy}
              onChange={(event: ChangeEvent<HTMLInputElement>) => {
                const file = event.target.files?.[0];
                if (file) void handleFile(file);
              }}
            />
          </label>
        ) : (
          <div className="asset-card">
            <div className="asset-icon"><FilmIcon /></div>
            <div className="asset-copy">
              <strong title={fileName}>{fileName}</strong>
              <span>
                {video.metadata.width} × {video.metadata.height} · {video.metadata.fps.toFixed(1)} fps
              </span>
            </div>
            <button
              className="icon-button"
              disabled={busy}
              onClick={() => void discardCurrentVideo()}
              title="Discard this video"
            >
              ×
            </button>
          </div>
        )}

        {video && selection && !job && (
          <div className="tool-group">
            <p className="tool-label">Mask correction</p>
            <div className="segmented-control">
              <button
                className={mode === "add" ? "selected" : ""}
                onClick={() => setMode("add")}
              >
                <span>＋</span> Add
              </button>
              <button
                className={mode === "subtract" ? "selected danger" : ""}
                onClick={() => setMode("subtract")}
              >
                <span>−</span> Subtract
              </button>
            </div>
            <p className="tool-hint">
              Click missing areas to add them. Switch to subtract and click spill areas.
            </p>
          </div>
        )}

        {video &&
          selection &&
          (!job || ["completed", "failed"].includes(job.status)) && (
          <div className="model-picker">
            <div className="model-picker-heading">
              <label htmlFor="tracking-model">Tracking model</label>
              <span>{selectedModel.parameters} parameters</span>
            </div>
            <select
              id="tracking-model"
              value={modelVariant}
              disabled={busy || job?.status === "queued" || job?.status === "running"}
              onChange={(event) =>
                setModelVariant(event.target.value as Sam2ModelVariant)
              }
            >
              {SAM2_MODEL_OPTIONS.map((model) => (
                <option value={model.variant} key={model.variant}>
                  SAM2.1 Hiera {model.label} — {model.profile}
                </option>
              ))}
            </select>
            <p>{selectedModel.description}</p>
          </div>
          )}

        {job && (
          <div className="render-card">
            <div className="render-row">
              <span>{job.status === "completed" ? "Render complete" : "Rotoscoping"}</span>
              <strong>{job.status === "completed" ? "100%" : `${progress}%`}</strong>
            </div>
            <div className="progress-track">
              <span style={{ width: `${job.status === "completed" ? 100 : progress}%` }} />
            </div>
            <small>
              {job.processed_frames} / {job.total_frames} frames · {job.status}
              {jobModel ? ` · ${jobModel.label}` : ""}
            </small>
            {job.status === "failed" && <p className="inline-error">{job.error}</p>}
          </div>
        )}

        <div className="panel-actions">
          {selection &&
            (!job || ["completed", "failed"].includes(job.status)) && (
            <button className="primary-button" disabled={busy} onClick={renderVideo}>
              {busy
                ? "Working…"
                : job
                  ? `Render again with ${selectedModel.label}`
                  : `Render with ${selectedModel.label}`}
              <ArrowIcon />
            </button>
            )}
          {job?.status === "completed" && job.output_url && (
            <a
              className="primary-button"
              href={`${API_BASE}${job.output_url}`}
              download
            >
              Download alpha video <DownloadIcon />
            </a>
          )}
          {job?.status === "completed" && (
            <div className="new-video-action">
              <button
                type="button"
                className="secondary-button"
                disabled={busy}
                onClick={() => void discardCurrentVideo()}
              >
                <span>{busy ? "Clearing project…" : "Add another video"}</span>
                <UploadIcon />
              </button>
              <small>
                Starts fresh and wipes the current upload, masks, cached frames,
                and render from FrameSeg.
              </small>
            </div>
          )}
        </div>
      </aside>

      <section className="viewer-panel">
        <div className="viewer-toolbar">
          <div className="viewer-title">
            <span className="status-dot" />
            <strong>Object matte</strong>
            <span>Frame {String(previewFrame).padStart(4, "0")}</span>
          </div>
          {video && (
            <div className="viewer-actions">
              <span className="zoom-badge">{Math.round(zoom * 100)}%</span>
              <button
                type="button"
                className="zoom-reset"
                onClick={resetViewport}
                disabled={zoom === MIN_ZOOM && pan.x === 0 && pan.y === 0}
              >
                Reset view
              </button>
              <span className="codec-badge">
                {video.metadata.codec.toUpperCase()} · {video.metadata.container.toUpperCase()}
              </span>
            </div>
          )}
        </div>

        <div
          ref={stageRef}
          className={`canvas-stage ${busy ? "canvas-busy" : ""} ${
            spacePressed && zoom > MIN_ZOOM ? "pan-ready" : ""
          } ${isPanning ? "is-panning" : ""}`}
          onPointerDown={onStagePointerDown}
          onPointerMove={onStagePointerMove}
          onPointerUp={finishPanning}
          onPointerCancel={finishPanning}
        >
          {video ? (
            <>
              <canvas
                ref={canvasRef}
                className={`selection-canvas ${renderLocked ? "canvas-locked" : ""}`}
                style={{
                  transform: `translate3d(${pan.x}px, ${pan.y}px, 0) scale(${zoom})`,
                }}
                onClick={onCanvasClick}
              />
              {!selection && !busy && (
                <div className="canvas-instruction">
                  <CursorIcon />
                  <span>Click the object you want to isolate</span>
                </div>
              )}
              {busy && (
                <div className="processing-overlay">
                  <span className="spinner" />
                  <strong>{selection ? "Updating mask" : "SAM2 is tracing the object"}</strong>
                </div>
              )}
            </>
          ) : (
            <div className="empty-viewer">
              <div className="empty-orbit"><FrameIcon /></div>
              <h3>Your subject appears here</h3>
              <p>Import a short clip to begin an AI-assisted rotoscope.</p>
            </div>
          )}
        </div>

        {video && !job && (
          <section className="timeline-panel" aria-label="Video frame range">
            <div className="timeline-heading">
              <div>
                <strong>Clip range</strong>
                <span>
                  Frames {clipStart}–{clipEnd} · {clipFrames} selected
                </span>
              </div>
              <span>
                {formatTime(clipStart / video.metadata.fps)}–
                {formatTime((clipStart + clipFrames) / video.metadata.fps)}
              </span>
            </div>

            <div
              ref={timelineRef}
              className={`filmstrip ${timelineDragMode ? "timeline-dragging" : ""}`}
              onPointerMove={onTimelinePointerMove}
              onPointerUp={finishTimelineDrag}
              onPointerCancel={finishTimelineDrag}
            >
              {timelineFrames.map((frameIndex) => (
                // Dynamic frame endpoints are already cached PNG thumbnails.
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  key={frameIndex}
                  src={`${assetUrl(video.first_frame_url)}?frame_index=${frameIndex}`}
                  alt=""
                  loading="lazy"
                />
              ))}
              <span
                className={`clip-window ${selection || busy ? "range-locked" : ""} ${
                  timelineDragMode === "move" ? "window-moving" : ""
                }`}
                style={{ left: `${selectionLeft}%`, width: `${selectionWidth}%` }}
                onPointerDown={(event) => beginTimelineDrag(event, "move")}
                title="Drag to move the selected range"
              >
                <i
                  onPointerDown={(event) =>
                    beginTimelineDrag(event, "resize-start")
                  }
                  title="Drag to change the start frame"
                />
                <i
                  onPointerDown={(event) =>
                    beginTimelineDrag(event, "resize-end")
                  }
                  title="Drag to change the end frame"
                />
              </span>
              <span
                className={`seed-marker ${selection || busy ? "seed-locked" : ""}`}
                style={{
                  left: `${((displayedSeedFrame + 0.5) / totalFrames) * 100}%`,
                }}
                onPointerDown={(event) => beginTimelineDrag(event, "seed")}
                title={`Selection frame ${displayedSeedFrame} · drag to reposition`}
              />
            </div>

            <div className="timeline-scale">
              <span>0:00</span>
              <span>{formatTime(totalFrames / video.metadata.fps)}</span>
            </div>

            {selection ? (
              <button
                className="timeline-edit-button"
                disabled={busy}
                onClick={editClipRange}
              >
                Change range and reset mask
              </button>
            ) : (
              <p className="timeline-hint">
                Drag the highlighted range to move it, drag either edge to resize it,
                and drag the white marker to choose the clearest selection frame.
              </p>
            )}
          </section>
        )}

        <div className="viewer-footer">
          <span><kbd>Click</kbd> select / correct</span>
          <span><kbd>Scroll</kbd> zoom</span>
          <span><kbd>Space</kbd> + drag pan</span>
          <span><i className="mask-swatch" /> Active mask</span>
          <span>{clipFrames} selected · maximum 1,200</span>
        </div>
      </section>

      {error && (
        <div className="error-toast" role="alert">
          <span>!</span>
          <p>{error}</p>
          <button onClick={() => setError(null)}>×</button>
        </div>
      )}
    </div>
  );
}

function loadImage(source: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Could not load the preview image"));
    image.src = source;
  });
}

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00.0";
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds - minutes * 60;
  return `${minutes}:${remainder.toFixed(1).padStart(4, "0")}`;
}

function UploadIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden><path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M5 14v4a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-4" /></svg>;
}
function FilmIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M7 4v16M17 4v16M3 9h4m10 0h4M3 15h4m10 0h4" /></svg>;
}
function ArrowIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden><path d="M5 12h14m-5-5 5 5-5 5" /></svg>;
}
function DownloadIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden><path d="M12 3v12m0 0 5-5m-5 5-5-5M5 21h14" /></svg>;
}
function CursorIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden><path d="m5 3 13 9-6 1-3 6L5 3Z" /></svg>;
}
function FrameIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden><path d="M8 3H5a2 2 0 0 0-2 2v3m13-5h3a2 2 0 0 1 2 2v3M8 21H5a2 2 0 0 1-2-2v-3m13 5h3a2 2 0 0 0 2-2v-3" /><circle cx="12" cy="12" r="3" /></svg>;
}
