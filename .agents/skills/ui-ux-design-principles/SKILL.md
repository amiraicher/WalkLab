---
name: ui-ux-design-principles
description: Mandatory UI/UX rules for creating intuitive, persona-focused interfaces in the WalkingLab app. Must be consulted before scaffolding new web pages, changing layouts, or adding controls.
---

# UI / UX Design Principles

## 1. Persona-First & Cognitive Load
- The role of each screen must be explicitly clear to the user persona (e.g., Clinical Researcher, Patient, Data Annotator).
- **Walk the User Through the App:** The UI must guide the user step-by-step. Use wizard-like flows, clear onboarding instructions, or logical top-to-bottom/left-to-right visual hierarchies.
- **Predictability:** What happens when a toggle or button is clicked must be immediately evident through clear labeling, tooltips, or state changes. Avoid hidden modes or unexpected screen jumps.

## 2. Layout & Contextual Real Estate
- **Centralize the Main Action:** For video/image tasks, the media viewer should be the centerpiece (the "hero" element).
- **Consolidated Controls:** Do not scatter related actions across multiple disjointed frames. 
  - *Example:* Instead of having one frame for a live camera feed and a completely separate frame for video upload, use a **single unified media frame** with a clear toggle (e.g., segmented control or tabs) to switch between `Live Camera` and `Upload File` modes.
- Keep the main viewport clean by placing attribute controls, keypoint lists, and metadata in a collapsible sidebar or organized control panel underneath.

## 3. Computer Vision & Annotation Specifics
- **Temporal Tracking & Controls:** For video tasks, provide a robust timeline/scrubber interface.
- **Skeleton Rigging & Overlay:** Modern dashboards use a visual "skeleton" connecting keypoints (e.g., shoulders to elbows) based on anatomical schemas, rather than just disconnected dots.
- **Handling Occlusions:** Provide a way to mark or visualize occluded/hidden points (e.g., dashed lines or faded colors).
- **Reduce Clicks:** Design keyboard shortcuts and auto-advancement where possible to minimize mouse fatigue during annotation.

## 4. Visual Feedback & State Management
- Never leave the user guessing if an action (like "uploading" or "running inference") is occurring. Use spinners, progress bars, and toast notifications.

## 5. Mandatory Agent Workflow: Web Search & Proposals
- **Crucial Rule:** Before implementing *any* major UI change or creating a new screen, the agent **MUST** search the web for modern UX/UI examples with similar functionalities.
- The agent must present these layout suggestions to the user for review and selection *before* writing the code. Do not assume a layout; provide the user with options inspired by real-world, best-in-class interfaces.
