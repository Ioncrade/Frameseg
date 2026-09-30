# FrameSeg backend optimization review

Date: 2026-09-29

## Implemented result

The immediate throughput, memory, and output-size work described below is now in the
backend. The production defaults are:

- `facebook/sam2.1-hiera-base-plus`, preloaded during application startup;
- CUDA-resident SAM2 temporal state, with CPU offload available as a configuration;
- bounded three-frame decode prefetch and three-frame encode queues;
- GPU-side sigmoid, mask union, and `uint8` alpha conversion;
- direct incremental alpha-video encoding with no PNG spool or second stitching pass;
- QTRLE MOV as the fast, lossless-alpha default, with ProRes 4444 still available;
- zero RGB data under fully transparent pixels; and
- explicit inference-session cleanup after selection and tracking.

Measured on the real 2560×1440, 60 fps source and accepted mask used in this review:

| Pipeline | 120-frame time | Throughput | 120-frame output |
| --- | ---: | ---: | ---: |
| Optimized ProRes + large model | 16.43 s | 7.31 fps | 14.1 MB |
| Optimized QTRLE + SAM2.1 base-plus | 9.86 s | 12.18 fps | 8.1 MB |

During the final QTRLE benchmark, GPU utilization averaged 36.5% and peaked at 53%;
peak GPU memory was approximately 3.37 GiB. A linear projection puts 1,200 frames at
about 99 seconds rather than the reported 200+ seconds. This is a projection from a
120-frame sample, not a substitute for a final 1,200-frame soak test.

QTRLE is intended as the fast alpha-MOV path. Set
`FRAMESEG_OUTPUT_CODEC=prores_4444` when a downstream editor specifically requires
ProRes 4444. QTRLE and ProRes round-trip tests both verify that the alpha plane is
present. The ordinary render path creates only the final MOV; debug PNG artifacts are
opt-in.

## Executive decision

The original backend was functionally streaming but not production-efficient. The
highest-value changes identified during the review were:

1. Encode the transparent video incrementally as each mask is produced. Do not write
   RGB, alpha, and RGBA PNG sequences during a normal render.
2. Zero the RGB channels wherever alpha is zero before encoding. The current output
   carries the complete, invisible source image under transparent pixels and this is
   the main reason the ProRes file is abnormally large.
3. Make the SAM2 inference-state device configurable. Keep the bounded temporal state
   on the GPU when VRAM permits; the current CPU offload transfers memory features back
   to the GPU on every frame.
4. Preload the model and reuse the already-generated first-frame preview for selection.
5. Add explicit session cleanup, abandoned-selection TTLs, job cancellation, artifact
   retention, and a single-GPU worker queue.

Do not call `torch.cuda.empty_cache()` for every frame. It will not fix live-tensor
retention and is likely to reduce throughput.

## Evidence from the current 1,200-frame job

The reviewed job uses the reported 130 MB source file. Exact probe results:

| Item | Observed value |
| --- | ---: |
| Source | H.264, 2560×1440, 60 fps, 38.25 s, 136.5 MB |
| Processed portion | 1,200 frames, approximately 20 s |
| Transparent output | ProRes 4444, 2560×1440, 60 fps |
| Output bitrate | approximately 1.294 Gbit/s |
| Output MOV | 3.236 GB |
| RGB PNG directory | 2.468 GB |
| RGBA PNG directory | 2.643 GB |
| Alpha PNG directory | 5.2 MB |
| Total job directory | 8.352 GB |

The output MOV is approximately 23.7 times the size of the full 38-second source,
even though only 20 seconds were processed. The complete render directory is roughly
61 times the source size.

The alpha PNGs are not the large disk consumer. They occupy only about 5.2 MB. The
unnecessary RGB and RGBA PNG copies consume about 5.1 GB, and the MOV consumes another
3.2 GB.

## Findings

### P0 — The output path performs triple PNG writes followed by a second full pass

`backend/mask_service.py:38-44` creates `rgb_frames`, `alpha_masks`, and
`rgba_frames` for every job. `write_frame()` then:

- creates a full-resolution alpha array;
- creates a second full-resolution RGBA array;
- compresses and writes RGB PNG;
- compresses and writes alpha PNG;
- compresses and writes RGBA PNG.

After all 1,200 frames have been processed, `finalize()` reopens and decodes every RGBA
PNG before ProRes encoding (`backend/mask_service.py:88-121`). This creates two serial
phases:

```text
decode -> SAM2 -> PNG compression × 3 -> disk
                                      then
disk -> PNG decompression -> ProRes encode -> MOV
```

This costs CPU time, filesystem operations, approximately 5.1 GB of avoidable disk,
and a long finalization delay after model tracking has already finished.

Recommendation: replace `MaskOutputService` with an incremental encoder that opens an
atomic `.partial` output at job start, accepts one RGB + alpha frame at a time, encodes
and muxes it immediately, and flushes/renames at completion. Debug PNG generation must
be an explicit opt-in setting, disabled by default.

As a transitional patch only, PNGs can be deleted after each one is consumed by
`finalize()`. That reduces post-render residue but does not reduce peak disk usage or
remove the second pass, so it is not the target architecture.

### P0 — Invisible RGB is being encoded

`backend/mask_service.py:66` creates RGBA with:

```python
rgba = np.dstack((rgb_array, alpha))
```

This preserves the original RGB values even where alpha is zero. Those colors are
invisible to the viewer, but the codec must still encode their full image detail.
ProRes 4444 is an intraframe 4:4:4 codec, so this hidden detail is extremely expensive.

For straight-alpha output, preserve RGB on visible and partially visible pixels but
zero fully transparent pixels:

```python
rgb_for_output = rgb_array.copy()
rgb_for_output[alpha <= 1] = 0
rgba = np.dstack((rgb_for_output, alpha))
```

The `<= 1` threshold should be quality-tested around hair and motion blur. A strict
`alpha == 0` condition is the safest initial behavior.

Do not premultiply RGB by alpha unless the file and downstream applications explicitly
agree that the result is premultiplied alpha. A straight/premultiplied mismatch creates
dark or bright edge halos.

### Measured output experiments

The first 120 frames (2 seconds) of the real 2560×1440 render were encoded locally:

| Encoding | 2-second result | Relative to current sample |
| --- | ---: | ---: |
| Current-style ProRes 4444, 16-bit alpha | 366.4 MB | 100% |
| ProRes 4444, 8-bit alpha only | 365.5 MB | 99.8% |
| ProRes 4444, 8-bit alpha, zero transparent RGB | 14.7 MB | 4.0% |
| Premultiplied ProRes 4444, 8-bit alpha | 13.1 MB | 3.6% |
| VP9 alpha WebM, CRF 18 | 9.7 MB | 2.7% |
| VP9 alpha WebM, CRF 24 | 7.4 MB | 2.0% |
| Separate HEVC grayscale matte, CRF 12 | 173 KB | 0.05% |

These are short-sample measurements, not guaranteed full-clip projections. They show
that clearing invisible RGB matters much more than changing ProRes alpha from 16 bits
to 8 bits. Based on this sample, an optimized 20-second ProRes file may be on the order
of hundreds rather than thousands of megabytes, but the final size will depend on how
much of the frame the subject occupies and how complex its visible pixels are.

### P0 — Codec policy should match the product use case

The current encoder uses `prores_ks`, profile 4 (ProRes 4444), and does not specify
`alpha_bits` (`backend/mask_service.py:99-107`). FFmpeg therefore uses the encoder's
16-bit alpha default even though FrameSeg generates an 8-bit mask.

Recommended export modes:

| Mode | Representation | Intended use |
| --- | --- | --- |
| `editing` | ProRes 4444 MOV, 8-bit alpha, zero hidden RGB | NLE/VFX compatibility |
| `delivery` | VP9 alpha WebM, quality-controlled CRF | Much smaller browser/delivery artifact |
| `matte` | Original compressed source plus grayscale mask video | Smallest internal/storage representation |

For `editing`, configure `alpha_bits=8` and benchmark `bits_per_mb` or fixed `qscale`
only after invisible RGB is cleared. Lowering the ProRes bit budget without edge-quality
tests can damage hair, motion blur, and semi-transparent boundaries.

For `delivery`, VP9 alpha was approximately 37–49 times smaller than the unmodified
ProRes sample. Validate browser, editor, and decode compatibility for the target users.
WebM is not a drop-in MOV replacement.

For the internal `matte` representation, keep the original H.264/H.265 upload and
encode only the grayscale mask. Composite source + mask in the UI or generate the
requested delivery/editing format on demand. A lossy matte codec requires aggressive
edge regression tests; FFV1 or another lossless grayscale codec is an alternative when
exact alpha values matter more than minimum size.

Ordinary H.264 and H.265 MP4 outputs do not provide a portable, conventional alpha
channel. Apple HEVC-with-alpha exists, but it is a specialized compatibility path and
should be a separate platform-specific export rather than the only backend format.

### P0 — The backend streams decoded frames; it does not retain all 1,200 RGB frames

The active render uses `iter_video_frames()` (`backend/media_loader.py:148-177`), which
yields one `FramePacket` at a time. The list-producing `load_video()` exists but is not
used by the API render path.

The SAM2 path also attempts bounded state:

- `session.processed_frames.pop(packet.index, None)` releases the current processed
  frame (`backend/model_service.py:310-312`);
- the vision feature cache is configured to one frame;
- `_prune_tracking_history()` retains only the temporal-memory/object-pointer window
  required for tracking (`backend/model_service.py:383-402`).

Therefore, a high value in `nvidia-smi` is not sufficient proof that 1,200 frames remain
live. PyTorch uses a CUDA caching allocator, so reserved memory commonly stays visible
after tensors are released. Measure both allocated and reserved memory.

The current path can still be made more deterministic:

1. Put tracking-session cleanup in a `finally` block.
2. Clear `processed_frames`, inputs, output dictionaries, frame history, and the vision
   cache explicitly at job completion, cancellation, and failure.
3. Delete the final local references to output tensors before the next job.
4. Call `torch.cuda.empty_cache()` only at an idle job boundary if returning reserved
   blocks to other processes is operationally useful. Never call it per frame.

Add these metrics at job start, every 60 frames, and job end:

```python
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()
torch.cuda.max_memory_allocated()
process_rss_bytes
```

The acceptance condition is that allocated memory plateaus after SAM2's rolling window
is warm. Reserved memory may remain at its high-water mark.

### P0 — Abandoned selection sessions are a real memory-retention path

`Sam2ModelService._selection_sessions` can retain up to 16 interactive inference
sessions. A session is removed only when the client accepts or explicitly cancels it.
Closing a browser tab, losing the network, or abandoning a project does not call the
delete endpoint reliably. Eventually the service can retain 16 stale sessions and
reject new selections.

Implement:

- `last_accessed_at` on each selection session;
- a short inactivity TTL, for example 10–20 minutes;
- periodic cleanup that explicitly resets and removes the session;
- cleanup when a video/project is deleted;
- an API heartbeat while the editor is open;
- per-user/session quotas.

The API's `videos`, `selections`, and `jobs` dictionaries also never expire. Those
records are small, but their files persist indefinitely. Add project ownership,
persistent job metadata, and artifact retention policies.

## Tracking throughput review

### P0 — CPU-offloaded temporal state causes per-frame PCIe transfers

`Sam2ModelService` defaults `offload_state_to_cpu=True`, so
`inference_state_device` is CPU (`backend/model_service.py:102-116`). The installed
Transformers implementation stores mask-memory features on that device, then moves the
rolling memories back to the inference device for every following frame.

This saves VRAM but can materially reduce frames per second. Because FrameSeg already
prunes tracking history, add a deployment option such as:

```text
FRAMESEG_INFERENCE_STATE_DEVICE=auto|cuda|cpu
```

- `cuda`: preferred performance mode when the measured peak fits with safety margin;
- `cpu`: low-VRAM mode;
- `auto`: select CUDA only when a configured free-VRAM threshold is met.

Benchmark both modes on the production GPU at 720p, 1080p, and 1440p. Record p50 frame
latency, GPU utilization, PCIe traffic, peak allocated VRAM, and system RSS.

### P0 — Model inference and PNG work are serialized

The model's global lock is held across generator yields in `track_frames()`. While the
consumer writes three PNG files, the lock remains held and the GPU is generally idle.
No other selection can use the model during that output work either.

Use a bounded pipeline:

```text
decoder/prefetch (CPU, queue 2–4)
        -> SAM2 sequential inference (GPU)
        -> alpha conversion + encoder (bounded queue 2–4)
```

SAM2 temporal propagation remains sequential, but CPU decode and output encoding can
overlap GPU inference. The queues must remain deliberately small so the optimization
does not recreate a frame-memory problem. Convert the mask to `uint8` before enqueueing
it; do not enqueue full-resolution float32 logits.

There should still be only one active inference job per GPU unless profiling proves
that safe concurrency improves throughput. Additional jobs should wait in an explicit
queue rather than starting background threads that block on the model lock.

### P1 — Full-resolution float masks cross to CPU unnecessarily

`_post_process_masks()` converts the full-resolution result to float32 CPU NumPy
(`backend/model_service.py:352-363`). `mask_logits_to_alpha()` then creates a CPU Torch
view, applies sigmoid and union operations, and produces `uint8`.

Perform sigmoid/union/clamp/quantization on the inference device and transfer only the
final `uint8` alpha plane to CPU. For one 2560×1440 object this reduces the mask transfer
from roughly 14.7 MB float32 to 3.7 MB uint8 per frame and removes CPU tensor work.

The tracking result should expose `alpha: np.ndarray[uint8]` instead of full-resolution
float logits unless a downstream consumer explicitly needs logits.

### P1 — First selection repeats work and pays cold-start latency

Upload currently probes the file and decodes/writes frame zero
(`backend/api.py:158-160`). Selection then probes the video and decodes frame zero again
(`backend/rotoscope_service.py:67-74`). Read the existing preview PNG or retain a
short-lived frame-zero cache and pass the existing `VideoMetadata`; do not reopen the
video for selection.

The first click also calls `self.load()` and loads the large SAM2 checkpoint lazily.
Move model loading to the FastAPI lifespan/startup phase, expose `loading` and `ready`
health states, and optionally run one warm-up inference. This moves the cost out of an
interactive request; it does not eliminate the cost.

The baseline model was `facebook/sam2-hiera-large`. Benchmark SAM2.1 tiny, small,
base-plus, and large on representative footage. A smaller model or a lower-resolution
tracking mode is the most direct latency/quality tradeoff. Make the model an environment
or deployment setting rather than a source-code constant.

### P2 — Compilation and hardware acceleration require measurement

Test `torch.compile` behind a feature flag after stage-level profiling. Compilation can
reduce Python/kernel-launch overhead, but it introduces warm-up latency, graph-break
risk, and potentially higher memory usage. SAM2's mutable inference session makes it
especially important to benchmark rather than assume a win.

Enable cuDNN autotuning only if inference shapes are stable. Investigate NVDEC or a
GPU-native decode/preprocess path after the larger PNG and CPU-offload bottlenecks have
been removed. Hardware encoding is useful for a separate H.264/H.265 matte, but it does
not replace a portable alpha-capable encoder.

## Production job architecture

FastAPI `BackgroundTasks` is not a durable GPU job system. A process restart loses job
state, there is no cancellation or admission control, and multiple renders can create
threads that wait on the same model lock.

Recommended architecture:

```text
FastAPI upload/control plane
        -> durable job record + explicit queue
        -> one worker per GPU
        -> bounded decode/inference/encode pipeline
        -> object storage / local artifact store
```

Minimum production controls:

- one admitted render per GPU and a visible queue position;
- cancellation checked between frames;
- job recovery or an explicit failed-on-restart state;
- upload, intermediate, and output quotas;
- cleanup on success, failure, and cancellation;
- output TTL and delete-project API;
- source deletion after the retention window;
- atomic partial output handling;
- stage timing and memory metrics;
- limits by duration, frame count, dimensions, and decoded pixel count.

## Recommended implementation sequence

### Phase 1 — Immediate output and cleanup fixes

1. Add stage timers and CUDA/RSS metrics to preserve a baseline.
2. Zero RGB where alpha is zero.
3. Set ProRes `alpha_bits=8`.
4. Replace PNG spooling with direct incremental encoding.
5. Keep debug PNGs behind `FRAMESEG_KEEP_FRAME_ARTIFACTS=false`.
6. Explicitly clear the SAM2 session in `finally`.
7. Add selection TTL and job artifact retention.

Expected effect: eliminate approximately 5.1 GB of intermediate files for the observed
job, eliminate the second stitching pass, sharply reduce ProRes size, and make cleanup
predictable.

### Phase 2 — Throughput

1. Reuse the uploaded first-frame preview and metadata.
2. Preload/warm SAM2 at startup.
3. Benchmark GPU vs CPU inference-state storage.
4. Convert logits to `uint8` alpha before CPU transfer.
5. Add bounded decode and encode stages around sequential SAM2 inference.
6. Benchmark SAM2.1 model sizes and an optional working-resolution cap.

### Phase 3 — Product formats and durable jobs

1. Add `editing`, `delivery`, and `matte` export profiles.
2. Add a durable single-GPU queue and cancellation.
3. Persist job/project state.
4. Move large artifacts to managed storage with TTLs.
5. Test `torch.compile`, NVDEC, and hardware matte encoding only after profiling.

## Acceptance tests

- A 1,200-frame job produces no RGB/RGBA/alpha directories by default.
- Output becomes downloadable immediately after the final encoder flush; there is no
  second 1,200-frame stitching phase.
- CUDA allocated memory plateaus after the temporal window warms up.
- Abandoned selections disappear after the configured TTL.
- Failed and cancelled jobs leave no `.partial` files or frame directories.
- Straight-alpha edges round-trip without dark/bright halos over black, white, and
  saturated backgrounds.
- Alpha quality is measured separately from RGB quality around hair, glass, blur, and
  fast motion.
- The current 2560×1440 test clip is benchmarked for total wall time, frames per second,
  peak RSS, peak allocated/reserved VRAM, output size, and cleanup time before and after
  each phase.
- Output compatibility is tested in the supported browsers and target NLE/VFX tools.

## References

- [Hugging Face SAM2 Video documentation](https://huggingface.co/docs/transformers/en/model_doc/sam2_video)
- [FFmpeg ProRes encoder options](https://ffmpeg.org/ffmpeg-codecs.html#ProRes)
- [Apple: HEVC Video with Alpha](https://developer.apple.com/videos/play/wwdc2019/506/)
- [PyTorch CUDA memory and `empty_cache`](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.empty_cache.html)
- [PyTorch `torch.compile` reference](https://docs.pytorch.org/docs/stable/generated/torch.compile)
