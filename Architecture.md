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
- **components/video_tracker.js**: Custom Vue 3 component that orchestrates both live webcam streaming and HTTP video file uploads, handling letterboxing calculations, precise ETA progress bars, and SVG marker synchronization.

### 2. Backend Package (`walkinglab.backend`)
Handles data processing and model inference. Exposes data (e.g., JSON coordinates) rather than rendered frames.
- **inference.py**: Handles OpenVINO CPU-accelerated model execution (`YOLO-Pose`), sharing a single base `ov.Model`, preprocessor (`preprocess`), and postprocessor (`postprocess`) across two dedicated, statically-shaped compilation pipelines (`_compile_pipeline`):
  - **Live Webcam (`realtime_compiled_model` / `infer` / `predict`)**: Compiled eagerly with static input shape `[1, 3, 640, 640]` (`REALTIME_BATCH_SIZE = 1`) and `ov.properties.hint.PerformanceMode.LATENCY` to spread single-frame execution across CPU cores for minimum latency.
  - **Video File (`batch_compiled_model` / `process_video_file`)**: Compiled lazily with static input shape `[4, 3, 640, 640]` (`BATCH_SIZE = 4`) and `ov.properties.hint.PerformanceMode.THROUGHPUT`, executed asynchronously via `ov.AsyncInferQueue`. Trailing partial batches (`< 4` frames) are zero-padded via `_pad_batch` to preserve the static input shape while unpacking only valid frames. Maintains empirical ETA data.
- **filtering.py**: Implements Kalman Filtering for signal stabilization and noise reduction of the raw trajectories.
- **analysis.py**: Calculates spatiotemporal parameters (stride, cadence) and kinematic curves from the smoothed trajectories.

## Data Flow
The system supports two distinct real-time pipelines:

### A. Live Webcam Pipeline
1. Frontend captures frames via HTML5 hidden canvas.
2. `video_tracker.js` emits Base64 image strings to NiceGUI (WebSockets).
3. Backend decodes the frame in `predict` and delegates to `infer`, executing synchronous `B=1` inference on the static `[1, 3, 640, 640]` `LATENCY`-optimized model.
4. JSON coordinates returned and mapped to an SVG overlay.

### B. File Upload Batch Pipeline
1. Frontend calculates exact duration and asks backend for an empirical ETA estimate (`/api/estimate`).
2. Video is uploaded to the backend via `XMLHttpRequest` (tracking upload bytes).
3. Backend streams the MP4 to disk (`shutil.copyfileobj`) avoiding RAM bloat.
4. Backend `AsyncInferQueue` batches frames (`B=4`, padding trailing partial batches via `_pad_batch`) for high-throughput OpenVINO CPU processing on the static `[4, 3, 640, 640]` `THROUGHPUT`-optimized model.
5. Frontend animates the "Analyzing pose..." bar using `requestAnimationFrame`.
6. JSON coordinates are returned in bulk and synced chronologically to the HTML5 video playback.
