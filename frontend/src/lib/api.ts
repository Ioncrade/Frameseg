export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ??
  "http://localhost:8000";

export type VideoMetadata = {
  container: string;
  codec: string;
  width: number;
  height: number;
  fps: number;
  duration_seconds: number | null;
  total_frames: number;
  min_clip_frames: number;
  max_frames: number;
};

export type VideoUpload = {
  video_id: string;
  first_frame_url: string;
  source_url: string;
  metadata: VideoMetadata;
};

export type Selection = {
  selection_id: string;
  alpha_url: string;
  revision: number;
  object_score: number;
  start_frame: number;
  seed_frame: number;
  frame_count: number;
};

export type Sam2ModelVariant = "tiny" | "small" | "base_plus" | "large";

export const SAM2_MODEL_OPTIONS: ReadonlyArray<{
  variant: Sam2ModelVariant;
  label: string;
  profile: string;
  parameters: string;
  description: string;
}> = [
  {
    variant: "tiny",
    label: "Tiny",
    profile: "Fastest",
    parameters: "39M",
    description: "Best for rapid previews and simple subjects.",
  },
  {
    variant: "small",
    label: "Small",
    profile: "Fast",
    parameters: "46M",
    description: "Fast tracking with a modest quality increase.",
  },
  {
    variant: "base_plus",
    label: "Base+",
    profile: "Balanced",
    parameters: "81M",
    description: "Recommended balance of edge quality and render time.",
  },
  {
    variant: "large",
    label: "Large",
    profile: "Highest quality",
    parameters: "224M",
    description: "Best masks for difficult motion, with slower rendering.",
  },
];

export type RenderJob = {
  job_id: string;
  model_variant: Sam2ModelVariant;
  start_frame: number;
  seed_frame: number;
  frame_count: number;
  status: "queued" | "running" | "completed" | "failed";
  processed_frames: number;
  total_frames: number;
  progress: number;
  error: string | null;
  output_url: string | null;
};

export function assetUrl(path: string): string {
  return path.startsWith("http") ? path : `${API_BASE}${path}`;
}

export async function uploadVideo(file: File): Promise<VideoUpload> {
  const body = new FormData();
  body.append("file", file);
  return request<VideoUpload>("/api/videos", { method: "POST", body });
}

export async function deleteVideo(videoId: string): Promise<void> {
  await request(`/api/videos/${videoId}`, { method: "DELETE" });
}

export async function createSelection(
  videoId: string,
  point: { x: number; y: number },
  startFrame: number,
  frameCount: number,
  seedFrame: number,
): Promise<Selection> {
  return request<Selection>("/api/selections", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      video_id: videoId,
      point,
      start_frame: startFrame,
      frame_count: frameCount,
      seed_frame: seedFrame,
    }),
  });
}

export async function refineSelection(
  selectionId: string,
  point: { x: number; y: number },
  mode: "add" | "subtract",
): Promise<Selection> {
  return request<Selection>(`/api/selections/${selectionId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      additions: mode === "add" ? [point] : [],
      subtractions: mode === "subtract" ? [point] : [],
    }),
  });
}

export async function cancelSelection(selectionId: string): Promise<void> {
  await request(`/api/selections/${selectionId}`, { method: "DELETE" });
}

export async function startRender(
  selectionId: string,
  modelVariant: Sam2ModelVariant,
): Promise<RenderJob> {
  return request<RenderJob>(`/api/selections/${selectionId}/render`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_variant: modelVariant,
    }),
  });
}

export async function getRenderJob(jobId: string): Promise<RenderJob> {
  return request<RenderJob>(`/api/jobs/${jobId}`);
}

async function request<T = unknown>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) message = payload.detail;
    } catch {
      // The fallback status message is sufficient for non-JSON errors.
    }
    throw new Error(message);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}
