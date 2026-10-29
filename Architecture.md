# WalkingLab Architecture

## Overview
The system uses a decoupled architecture to isolate the Python backend (inference, filtering, metrics) from the frontend (NiceGUI UI). The backend processes video data to extract keypoints, smooths trajectories, and calculates metrics. The frontend serves as a client to display results, modify annotations, and view trends.

## Modules

### 0. Package Root (`walkinglab`)
- **__init__.py**: Performs platform initialization. On Windows (`win32`), registers the virtual environment's root (`sys.prefix`) and `Scripts` directory via `os.add_dll_directory` and `PATH` so bundled `msvc-runtime` DLLs (`msvcp140.dll`, `vcruntime140.dll`, etc.) are resolved by OpenVINO (`_pyopenvino`) and PyTorch (`c10.dll`) even when launched through `uv`'s wrapper executable.

### 1. UI Package (`walkinglab.ui`)
Built with NiceGUI, leveraging Vue 3 custom components for high-performance video overlays.
- **app.py**: Main entry point for the NiceGUI app, manages tabs and API routes (e.g. `/api/upload_video`).
- **annotation.py**: Interface for manual keypoint correction and frame extraction.
- **pose.py**: Visualizer for the extracted pose keypoints over video frames.
- **settings.py**: Controls for loading videos, connecting cameras, and tuning filter hyperparameters.
- **metrics.py**: Dashboard for longitudinal trend analysis and kinematic curves.
- **components/video_tracker.py & components/video_tracker.js**: Custom Vue 3 component (`VideoTracker`) that orchestrates both live webcam streaming and HTTP video file uploads. In live webcam mode, maintains a tunable **200 ms delayed frame buffer** rendered onto a dedicated `<canvas ref="displayCanvas">`, assigns a sequential `frame_index` to every captured frame, dilates inference dispatch (default `1` in `4` frames), interpolates marker coordinates across intermediate frames, and dynamically expands the buffer delay (with console warnings) if backend inference latency exceeds half the buffer duration.

### 2. Backend Package (`walkinglab.backend`)
Handles data processing and model inference. Exposes data (e.g., JSON coordinates) rather than rendered frames.
- **inference.py**: Handles OpenVINO CPU-accelerated model execution (`YOLO-Pose`), sharing a single base `ov.Model`, preprocessor (`preprocess`), and postprocessor (`postprocess` — outputting normalized `{"id": int, "x": float, "y": float}` keypoints) across two dedicated, statically-shaped compilation pipelines (`_compile_pipeline`):
  - **Live Webcam (`realtime_compiled_model` / `infer` / `predict` / `process_frame`)**: Compiled eagerly with static input shape `[1, 3, 640, 640]` (`REALTIME_BATCH_SIZE = 1`) and `ov.properties.hint.PerformanceMode.LATENCY`. `process_frame` supports both legacy raw Base64 strings and indexed frame payloads (`{"frame_index": int, "frame_data": str}`), associating returning detections with their original `frame_index`, computing intermediate frame markers via `RealtimeFrameBufferInterpolator`, and monitoring inference duration against the buffer budget.
  - **Video File (`batch_compiled_model` / `process_video_file`)**: Compiled lazily with static input shape `[4, 3, 640, 640]` (`BATCH_SIZE = 4`) and `ov.properties.hint.PerformanceMode.THROUGHPUT`, executed asynchronously via `ov.AsyncInferQueue`. Trailing partial batches (`< 4` frames) are zero-padded via `_pad_batch` to preserve the static input shape while unpacking only valid frames. Maintains empirical ETA data.
- **filtering.py**: Implements NumPy-based (`np.interp`) real-time frame interpolation (`interpolate_frame_sequence`) and the stateful `RealtimeFrameBufferInterpolator` (keeping `smooth_trajectory` as a placeholder stub since offline batch mode infers every frame directly):
  - Supports tunable interpolation strategies (`"linear"`, `"kalman"`, `"none"`). For each marker `id`, constructs an index vector and interpolates `(x, y)` over `range(index.min(), index.max() + 1)` via `np.interp`.
  - Manages real-time frame dilation (`DEFAULT_DILATION_FACTOR = 4`) and display buffer timing (`DEFAULT_BUFFER_DELAY_MS = 200`).
  - **Adaptive Buffer Scaling (`check_and_adjust_buffer`)**: Automatically checks whether single-frame inference latency exceeds half the current buffer duration ($T_{\text{inference}} > 0.5 \cdot T_{\text{buffer}}$). When exceeded, logs a `WARNING` and dynamically expands `buffer_delay_ms` ($\lceil 2.2 \times T_{\text{inference}} \rceil$, up to `MAX_BUFFER_DELAY_MS = 2000`) so two inference keyframes always fit inside the display delay window.
- **analysis.py**: Calculates spatiotemporal parameters (stride, cadence) and kinematic curves from the smoothed trajectories.

## Data Flow
The system supports two distinct real-time pipelines:

### A. Live Webcam Pipeline (Buffered & Interpolated)
1. Frontend captures camera frames at native FPS into a FIFO `frameBuffer` with a monotonically increasing `frame_index` and timestamp.
2. Frontend dilates inference requests (`frameDilation = 4` by default), emitting `frame_ready` (`{ frame_index, frame_data }`) over NiceGUI WebSockets only on dilated keyframes while the worker is free.
3. Backend `process_frame` decodes the frame, runs synchronous `B=1` `LATENCY` inference, and passes `(frame_index, markers, inference_time_ms)` to `RealtimeFrameBufferInterpolator.record_inference`.
4. Backend interpolates all intermediate frames `[prev_frame_index .. frame_index]` using the configured tunable method (`"linear"` by default, or `"kalman"`), checks if `inference_time_ms > 0.5 * buffer_delay_ms` (logging a warning and expanding `buffer_delay_ms` if needed), and returns the indexed payload.
5. Frontend `displayCanvas` renders buffered video frames delayed by `bufferDelayMs` (`200 ms` default) and overlays the exact frame-synchronized markers (inferred or linearly interpolated between bracketing keyframes) on the SVG layer without lag.

### B. File Upload Batch Pipeline
1. Frontend calculates exact duration and asks backend for an empirical ETA estimate (`/api/estimate`).
2. Video is uploaded to the backend via `XMLHttpRequest` (tracking upload bytes).
3. Backend streams the MP4 to disk (`shutil.copyfileobj`) avoiding RAM bloat.
4. Backend `AsyncInferQueue` batches frames (`B=4`, padding trailing partial batches via `_pad_batch`) for high-throughput OpenVINO CPU processing on the static `[4, 3, 640, 640]` `THROUGHPUT`-optimized model.
5. Frontend animates the "Analyzing pose..." bar using `requestAnimationFrame`.
6. JSON coordinates are returned in bulk and synced chronologically to the HTML5 video playback.
