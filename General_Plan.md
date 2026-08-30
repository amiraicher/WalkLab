# Vision-Based Gait Tracking & Quantification Pipeline: System Architecture

## 1. Overview & Objectives
* **Goal:** Build a lightweight, semi-automated pipeline to track, quantify, and trend walking patterns from unconstrained smartphone videos (e.g., tripod setups at home) to objectively assess longitudinal changes (improvement vs. regression).
* **Approach:** 
  1. Lightweight keypoint extraction using an existing model.
  2. Geometric/anthropometric normalization to handle non-sagittal viewing angles using facial and upper-body anchors.
  3. Signal stabilization and noise reduction via **Kalman Filtering**.
  4. Custom annotation toolkit to bootstrap a dataset for future fine-tuning.

---

## 2. Module 1: Annotation Toolkit & Dataset Bootstrapping
* **Purpose:** Create a custom labeling and review interface to curate high-quality training subsets from your extensive video library.
* **Core Components:**
  * **Video Frame Extractor:** Slices raw videos into synchronized frames, allowing clipping of complete gait cycles (heel-strike to heel-strike).
  * **Visualizer & Corrector:** Renders baseline keypoints from a pre-trained model over video frames, allowing manual correction ("drag-and-drop") of drifted joints (especially ankles and feet during occlusion).
  * **Metadata Logging:** Tags sequences by date, camera angle/position, and qualitative observations to build a structured dataset for future fine-tuning.

---

## 3. Module 2: Pose Estimation & Viewpoint Normalization
* **Lightweight Model Selection:** Utilize efficient, real-time pose estimators such as **MediaPipe BlazePose** or **YOLO-Pose (Nano/Small variants)** to extract 2D joint coordinates ($x, y$) across time.
* **Handling Unconstrained Angles (Non-Sagittal Correction):**
  * Because casual videos are rarely strictly orthogonal (pure sagittal plane), raw joint coordinates suffer from perspective distortion and foreshortening.
  * **Anthropometric Anchoring:** Use invariant facial and upper body landmarks (e.g., distance between eyes, nose-to-ear ratios, or bi-acromial shoulder width) as fixed scaling factors within the frame.
  * **Affine/Homography Transformation:** Map the user's orientation relative to the camera frame by tracking the torso and head orientation vector, applying geometric projection to approximate standard planar coordinates or calculate relative scaling ratios immune to camera distance shifts.

---

## 4. Module 3: Trajectory Smoothing via Kalman Filtering
* **Purpose:** Raw keypoint outputs from frame-by-frame inference suffer from jitter, occlusion-induced spikes, and high-frequency noise. A **Kalman Filter** provides optimal state estimation.
* **Filter Design:**
  * **State Vector:** Modeled for each keypoint as position and velocity $[p_x, p_y, v_x, v_y]^T$.
  * **Transition Matrix ($F$):** Constant velocity motion model assuming smooth kinematic transitions between frames.
  * **Measurement Matrix ($H$):** Maps the noisy 2D coordinates observed by the pose estimation model.
  * **Process & Measurement Noise ($Q, R$):** Tuned to penalize sudden, unrealistic jumps (e.g., a foot keypoint teleporting 20 pixels in a single frame due to occlusion) while preserving genuine movement dynamics.

---

## 5. Module 4: Metric Extraction & Trend Analysis
* **Spatiotemporal Parameters:** Automatically segment gait cycles using smoothed ankle/heel trajectories to calculate:
  * Stride and step length (normalized by body height/facial scale).
  * Cadence (steps per minute).
  * Stance-to-swing phase ratio.
* **Kinematic Curves:** Generate normalized 2D angular trajectories (e.g., knee flexion/extension over a percentage of the gait cycle).
* **Longitudinal Dashboard:** Plot extracted metrics over weeks and months to visualize trends, highlighting variance and stability rather than absolute clinical diagnostic values.

---

## Architecture Pattern: Decoupled Inference & Client-Side Rendering
* **Backend (Python):** Acts as an inference and processing engine. It runs the pose estimation model, processes trajectories through the Kalman Filter, and serves lightweight coordinate data (`JSON`) rather than heavy rendered video frames.
* **Frontend (Web UI via NiceGUI / HTML5 Canvas):** Receives raw coordinate streams from the backend, renders the skeleton overlay dynamically on an HTML5 Canvas, and manages client-side interactivity (such as low-latency drag-and-drop keypoint corrections).


## Suggested Tech Stack

* **Pose Estimation:** `MediaPipe` or `Ultralytics YOLO`
* **Filtering:** `FilterPy` or custom NumPy-based Kalman Filter implementation