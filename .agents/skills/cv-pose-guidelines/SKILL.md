---
name: cv-pose-guidelines
description: Architectural rules and data conventions for integrating pose estimation models and handling non-sagittal camera perspectives. Use when designing video processing pipelines or mapping joint coordinates.
---

# CV Pose Estimation Guidelines

## Core Principles
1. **Headless Processing:** The backend model must output pure structured numerical data (JSON coordinate arrays), never pre-rendered video frames with drawings.
2. **Coordinate Normalization:** Because home videos vary in distance and angle, raw pixel coordinates must always be normalized relative to stable upper-body anatomical anchors (e.g., eye-to-eye distance or shoulder width) to ensure scale invariance.
3. **Non-Sagittal Handling:** When analyzing non-orthogonal camera angles, flag confidence degradation on rotational axes and apply affine projection scaling guidelines rather than assuming a pure 2D sagittal plane.

## Output Schema Standard
Every pose-inference step must conform to this dictionary format:
```json
{
  "frame_idx": 0,
  "timestamp_ms": 0.0,
  "keypoints": {
    "left_ankle": {"x": 0.0, "y": 0.0, "visibility": 0.0}
  }
}