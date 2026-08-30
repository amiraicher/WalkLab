# WalkingLab Architecture

## Overview
The system uses a decoupled architecture to isolate the Python backend (inference, filtering, metrics) from the frontend (NiceGUI UI). The backend processes video data to extract keypoints, smooths trajectories, and calculates metrics. The frontend serves as a client to display results, modify annotations, and view trends.

## Modules

### 1. UI Package (`walkinglab.ui`)
Built with NiceGUI. Handles the web-based interface and presentation layer.
- **app.py**: Main entry point for the NiceGUI app, manages tabs.
- **annotation.py**: Interface for manual keypoint correction and frame extraction.
- **pose.py**: Visualizer for the extracted pose keypoints over video frames.
- **settings.py**: Controls for loading videos, connecting cameras, and tuning filter hyperparameters.
- **metrics.py**: Dashboard for longitudinal trend analysis and kinematic curves.

### 2. Backend Package (`walkinglab.backend`)
Handles data processing and model inference. Exposes data (e.g., JSON coordinates) rather than rendered frames.
- **inference.py**: Handles Ultralytics YOLO-Pose model execution and baseline keypoint extraction. Performs geometric/anthropometric normalization.
- **filtering.py**: Implements Kalman Filtering for signal stabilization and noise reduction of the raw trajectories.
- **analysis.py**: Calculates spatiotemporal parameters (stride, cadence) and kinematic curves from the smoothed trajectories.

## Data Flow
1. Video/Camera Input -> Backend Inference (`YOLO-Pose`).
2. Raw Keypoints -> Backend Filtering (`Kalman Filter`).
3. Smoothed Trajectories -> Backend Analysis (Metrics).
4. Backend -> Frontend (via JSON/coordinates for rendering on HTML Canvas).
5. Frontend User Corrections -> Backend (updating the bootstrapped dataset).
