---

### Skill 2: `kalman-filtering-protocol`
**Path:** `skills/kalman-gait-guidelines/SKILL.md`

```markdown
---
name: kalman-gait-guidelines
description: Behavioral protocols and parameter constraints for applying Kalman filtering to noisy kinematic trajectories. Use when writing or tuning trajectory smoothing and gait segmentation logic.
---

# Kalman Filtering Protocol for Gait Trajectories

## Core Principles
1. **State Space Integrity:** Every tracked keypoint must maintain an independent state vector tracking position and velocity $[p_x, p_y, v_x, v_y]^T$.
2. **Preventing Phase Lag:** Tuning of Process Noise ($Q$) and Measurement Noise ($R$) must prioritize minimal temporal phase lag. Over-smoothing will distort abrupt kinematic events like heel-strike.
3. **Occlusion & Jitter Handling:** If a keypoint's model confidence drops below a safety threshold (e.g., visibility < 0.5), rely on the state prediction step rather than corrupting the trajectory with erratic measurements.

## Execution Workflow
1. Load raw coordinate time-series.
2. Initialize constant-velocity matrices per joint.
3. Run forward prediction-update loops.
4. Output smoothed coordinate arrays alongside velocity derivatives for gait event detection.