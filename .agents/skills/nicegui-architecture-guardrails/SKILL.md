---
name: nicegui-architecture-guardrails
description: Strict boundary rules defining Frontend vs. Backend responsibilities and client-server communication in NiceGUI apps. Use before modifying UI components, layout structures, or event handling.
---

# NiceGUI Architecture & Responsibility Guardrails

## 1. Separation of Responsibilities
* **Backend (Python Server):** 
  * Responsible *only* for crunching data: running pose inference, executing Kalman filters, calculating metrics, and managing file assets.
  * Must never render graphical overlays onto images (`cv2.putText`, `cv2.circle`) for UI streaming.
* **Frontend (Browser / HTML5 Canvas):** 
  * Responsible *only* for presentation and real-time interaction.
  * Receives lightweight JSON coordinates from the server and handles rendering skeletons dynamically via client-side Canvas/SVG.
  * Manages zero-latency local user interactions (e.g., manual keypoint drag-and-drop adjustments) locally before pushing confirmation back to the backend.

## 2. Communication Protocol
* Data transfer between backend state and frontend views must rely on **lightweight JSON payloads or WebSockets**, avoiding heavy media streaming where possible.

## 3. Testing Client-Server Synchronization
When implementing or modifying UI changes, ensure the following testing steps are addressed:
1. **State Isolation Check:** Verify that modifying UI inputs (like scrubbing video frames) triggers instant client-side updates without blocking the backend worker thread.
2. **Payload Validation:** Confirm that manual keypoint edits on the canvas correctly serialize back into the exact JSON schema expected by the backend storage engine.
3. **Reactivity Test:** Ensure state updates from long-running backend tasks (like running a batch filter) correctly propagate to UI status notifications.