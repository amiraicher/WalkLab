"""
Annotation Toolkit UI module for interactive COCO 17-keypoint editing, client-side
ground-truth session curation, browser-native session ZIP save/load, custom model weight
selection, and on-demand few-shot YOLO-Pose fine-tuning.
"""
import base64
import json
from pathlib import Path
import tempfile

from nicegui import run, ui

import walkinglab.backend.inference as inference
from walkinglab.backend.keypoints import (
    COCO_KEYPOINTS,
    KEYPOINT_BY_ID,
    build_yolo_pose_dataset_zip,
    format_annotations_for_export,
)
from walkinglab.backend.training import (
    DEFAULT_FEW_SHOT_CONFIG,
    DEFAULT_WEIGHTS_PATH,
    pack_session_to_zip,
    run_few_shot_training,
    unpack_session_from_zip,
)
from walkinglab.ui.components.video_tracker import (
    VALID_KEYPOINT_FRAME_STATES,
    VideoTracker,
)


class AnnotationController:
    """
    Controller managing keypoint visibility, active keypoint selection, lightweight
    synchronization with the frontend ground-truth session, model weights selection,
    and on-demand few-shot training.

    Parameters
    ----------
    tracker : VideoTracker, optional
        Bound ``VideoTracker`` component instance, by default None.
    """

    def __init__(self, tracker: VideoTracker | None = None) -> None:
        self.tracker: VideoTracker | None = tracker
        self.visibility_state: dict[int, bool] = {
            kp["id"]: True for kp in COCO_KEYPOINTS
        }
        self.active_keypoint_id: int | None = None
        self.current_frame_index: int = 0
        self.all_frames_markers: list[list[dict]] = []
        self.frame_images: dict[int, str] = {}
        self.video_name: str = "annotations"

        # Lightweight frontend session status & headless fallback store
        self.session_samples: list[dict] = []
        self.session_count: int = 0
        self.saved_sample_ids: list[str] = []
        self.current_session_markers: list[dict] = []
        self.active_session_index: int = -1
        self.in_session_review: bool = False

        # Model weights & few-shot learning state
        self.trained_weights_bytes: bytes | None = None
        self.trained_weights_filename: str = "yolo_pose_finetuned.pt"
        self.few_shot_epochs: int = int(DEFAULT_FEW_SHOT_CONFIG["epochs"])
        self.few_shot_lr0: float = float(DEFAULT_FEW_SHOT_CONFIG["lr0"])
        self.few_shot_freeze_layers: int = int(
            DEFAULT_FEW_SHOT_CONFIG["freeze_layers"]
        )

    @property
    def active_weights_path(self) -> str:
        """
        Retrieves the active YOLO-Pose model weights path from ``inference.config``.

        Returns
        -------
        str
            The currently configured model weights path.
        """
        return inference.config.active_weights_path

    @active_weights_path.setter
    def active_weights_path(self, weights_path: str | None) -> None:
        """
        Sets the active YOLO-Pose model weights path on ``inference.config`` without
        loading the model upfront.

        Parameters
        ----------
        weights_path : str or None
            Filesystem path to a custom ``.pt`` weights file, or ``None`` for default.
        """
        inference.config.active_weights_path = weights_path

    @property
    def total_session_frames(self) -> int:
        """
        Returns the number of ground-truth frames currently saved in the session.

        Returns
        -------
        int
            Count of saved session frames on the frontend (or local store in headless tests).
        """
        return max(self.session_count, len(self.session_samples))

    def bind_tracker(self, tracker: VideoTracker) -> None:
        """
        Binds a ``VideoTracker`` instance to this controller.

        Parameters
        ----------
        tracker : VideoTracker
            The frontend video tracker element to synchronize with.
        """
        self.tracker = tracker

    def set_model_weights(self, weights_path: str | None = None) -> str:
        """
        Configures the active YOLO-Pose model weights path for inference and few-shot learning
        without loading the model upfront.

        Parameters
        ----------
        weights_path : str or None, optional
            Filesystem path to a custom ``.pt`` weights file, or ``None`` to restore
            the default ``"yolo26s-pose.pt"``, by default None.

        Returns
        -------
        str
            The newly configured active weights path.
        """
        self.active_weights_path = weights_path
        return self.active_weights_path

    def _ensure_frame_slot(self, frame_index: int) -> list[dict]:
        """
        Ensures ``all_frames_markers`` has a list entry at ``frame_index``.

        Parameters
        ----------
        frame_index : int
            Zero-based frame index.

        Returns
        -------
        list of dict
            The mutable marker list for ``frame_index``.
        """
        target_idx = max(0, int(frame_index))
        while len(self.all_frames_markers) <= target_idx:
            self.all_frames_markers.append([])
        if not isinstance(self.all_frames_markers[target_idx], list):
            self.all_frames_markers[target_idx] = []
        return self.all_frames_markers[target_idx]

    def get_frame_marker(
        self, keypoint_id: int, frame_index: int | None = None
    ) -> dict | None:
        """
        Retrieves the marker dictionary for ``keypoint_id`` in the active session sample
        or the specified/current video frame.

        Parameters
        ----------
        keypoint_id : int
            The COCO keypoint index (0 through 16).
        frame_index : int, optional
            Frame index to query; defaults to ``self.current_frame_index``.

        Returns
        -------
        dict or None
            Marker dictionary if present, otherwise ``None``.
        """
        if frame_index is None and self.in_session_review:
            if 0 <= self.active_session_index < len(self.session_samples):
                sample_markers = (
                    self.session_samples[self.active_session_index].get(
                        "markers"
                    )
                    or []
                )
            else:
                sample_markers = self.current_session_markers or []
            for marker in sample_markers:
                if isinstance(marker, dict) and int(
                    marker.get("id", -1)
                ) == int(keypoint_id):
                    return marker
            return None

        idx = (
            self.current_frame_index if frame_index is None else int(frame_index)
        )
        if idx < 0 or idx >= len(self.all_frames_markers):
            return None
        for marker in self.all_frames_markers[idx] or []:
            if isinstance(marker, dict) and int(marker.get("id", -1)) == int(
                keypoint_id
            ):
                return marker
        return None

    def toggle_keypoint_visibility(self, keypoint_id: int) -> bool:
        """
        Flips the overlay visibility boolean for ``keypoint_id`` and notifies the tracker.

        Parameters
        ----------
        keypoint_id : int
            The COCO keypoint index (0 through 16).

        Returns
        -------
        bool
            The updated visibility state (True if visible, False if hidden).
        """
        kp_id = int(keypoint_id)
        current_val = self.visibility_state.get(kp_id, True)
        new_val = not current_val
        self.visibility_state[kp_id] = new_val
        if self.tracker is not None:
            self.tracker.set_keypoint_visibility(kp_id, new_val)
        return new_val

    def set_all_visibility(self, visible: bool) -> None:
        """
        Sets the overlay visibility of all 17 COCO keypoints simultaneously.

        Parameters
        ----------
        visible : bool
            True to show all keypoints, False to hide all keypoints.
        """
        vis_bool = bool(visible)
        for kp in COCO_KEYPOINTS:
            self.visibility_state[kp["id"]] = vis_bool
        if self.tracker is not None:
            self.tracker.set_all_keypoints_visibility(vis_bool)

    def select_keypoint_for_edit(self, keypoint_id: int) -> int | None:
        """
        Toggles interactive edit mode for ``keypoint_id`` on the video canvas.

        Parameters
        ----------
        keypoint_id : int
            The COCO keypoint index (0 through 16) to toggle.

        Returns
        -------
        int or None
            The newly active keypoint ID, or ``None`` if deactivated.
        """
        kp_id = int(keypoint_id)
        if self.active_keypoint_id == kp_id:
            self.active_keypoint_id = None
        else:
            self.active_keypoint_id = kp_id

        if self.tracker is not None:
            self.tracker.set_active_keypoint(self.active_keypoint_id)
        return self.active_keypoint_id

    def set_keypoint_frame_state(self, keypoint_id: int, state: str) -> None:
        """
        Updates the state of ``keypoint_id`` in the active session sample or current frame
        and notifies the tracker.

        Parameters
        ----------
        keypoint_id : int
            The COCO keypoint index (0 through 16).
        state : str
            Target state: ``"visible"``, ``"occluded"``, or ``"delete"``.

        Raises
        ------
        ValueError
            If ``state`` is not one of ``("visible", "occluded", "delete")``.
        """
        if state not in VALID_KEYPOINT_FRAME_STATES:
            raise ValueError(
                f"Invalid keypoint frame state '{state}'. Must be one of {VALID_KEYPOINT_FRAME_STATES}."
            )

        kp_id = int(keypoint_id)
        if self.in_session_review:
            if 0 <= self.active_session_index < len(self.session_samples):
                frame_markers = self.session_samples[
                    self.active_session_index
                ].setdefault("markers", [])
            else:
                frame_markers = self.current_session_markers
        else:
            frame_markers = self._ensure_frame_slot(self.current_frame_index)

        existing_idx = next(
            (
                i
                for i, m in enumerate(frame_markers)
                if isinstance(m, dict) and int(m.get("id", -1)) == kp_id
            ),
            None,
        )

        if state == "delete":
            if existing_idx is not None:
                frame_markers.pop(existing_idx)
        elif state == "occluded":
            if existing_idx is not None:
                frame_markers[existing_idx]["visibility"] = 0.0
            else:
                frame_markers.append(
                    {"id": kp_id, "x": 0.5, "y": 0.5, "visibility": 0.0}
                )
        elif state == "visible":
            if existing_idx is not None:
                frame_markers[existing_idx]["visibility"] = 1.0
            else:
                frame_markers.append(
                    {"id": kp_id, "x": 0.5, "y": 0.5, "visibility": 1.0}
                )

        if self.tracker is not None:
            self.tracker.set_frame_keypoint_state(kp_id, state)

    def save_current_frame_to_session(
        self, image_b64: str | None = None
    ) -> dict:
        """
        Triggers client-side session frame capture on ``self.tracker`` (and updates
        ``session_samples`` when invoked in headless tests).

        Parameters
        ----------
        image_b64 : str or None, optional
            Optional base64 frame image for headless test usage, by default None.

        Returns
        -------
        dict
            Metadata dictionary for the saved sample.
        """
        frame_idx = max(0, int(self.current_frame_index))
        video_stem = Path(self.video_name or "video").stem or "video"
        sample_id = f"{video_stem}_frame_{frame_idx:06d}"

        resolved_image = image_b64 or self.frame_images.get(frame_idx, "")
        if (
            self.in_session_review
            and 0 <= self.active_session_index < len(self.session_samples)
        ):
            current_sample = self.session_samples[self.active_session_index]
            raw_markers = current_sample.get("markers") or []
            if not resolved_image:
                resolved_image = current_sample.get("image_b64", "")
        else:
            raw_markers = self._ensure_frame_slot(frame_idx)

        cloned_markers = [
            {
                "id": int(m["id"]),
                "x": float(m.get("x", 0.0)),
                "y": float(m.get("y", 0.0)),
                "visibility": float(m.get("visibility", 1.0)),
            }
            for m in raw_markers
            if isinstance(m, dict) and "id" in m
        ]

        sample_obj = {
            "sample_id": sample_id,
            "video_name": video_stem,
            "frame_index": frame_idx,
            "image_b64": resolved_image,
            "markers": cloned_markers,
        }

        existing_idx = next(
            (
                i
                for i, s in enumerate(self.session_samples)
                if s.get("sample_id") == sample_id
            ),
            None,
        )
        if existing_idx is not None:
            self.session_samples[existing_idx] = sample_obj
            self.active_session_index = existing_idx
        else:
            self.session_samples.append(sample_obj)
            self.active_session_index = len(self.session_samples) - 1

        if sample_id not in self.saved_sample_ids:
            self.saved_sample_ids.append(sample_id)
        self.session_count = max(self.session_count, len(self.saved_sample_ids))

        if self.tracker is not None:
            self.tracker.save_current_frame_to_session()

        return sample_obj

    def select_session_sample(self, index: int) -> dict | None:
        """
        Switches into session review mode and selects the ground-truth sample at ``index``.

        Parameters
        ----------
        index : int
            Zero-based index in the session.

        Returns
        -------
        dict or None
            The selected sample dictionary if available in local store, else ``None``.
        """
        idx = int(index)
        total = self.total_session_frames
        if idx < 0 or idx >= total:
            return None

        self.active_session_index = idx
        self.in_session_review = True
        sample = None
        if idx < len(self.session_samples):
            sample = self.session_samples[idx]
            self.current_frame_index = int(sample.get("frame_index", 0))

        if self.tracker is not None:
            self.tracker.select_session_sample(idx)
        return sample

    def step_session_sample(self, delta: int) -> dict | None:
        """
        Cycles forward or backward through saved session samples by ``delta``.

        Parameters
        ----------
        delta : int
            Step offset (e.g., ``-1`` for previous sample, ``1`` for next sample).

        Returns
        -------
        dict or None
            The newly selected sample dictionary if available in local store, else ``None``.
        """
        total = self.total_session_frames
        if total <= 0:
            return None

        base_idx = (
            self.active_session_index if self.active_session_index >= 0 else 0
        )
        next_idx = (base_idx + int(delta)) % total
        self.active_session_index = next_idx
        self.in_session_review = True
        sample = None
        if next_idx < len(self.session_samples):
            sample = self.session_samples[next_idx]
            self.current_frame_index = int(sample.get("frame_index", 0))

        if self.tracker is not None:
            self.tracker.step_session_sample(int(delta))
        return sample

    def delete_session_sample(self, index: int | None = None) -> None:
        """
        Removes a sample from the session and synchronizes with the frontend tracker.

        Parameters
        ----------
        index : int or None, optional
            Zero-based sample index to delete, or ``None`` to delete ``active_session_index``.
        """
        total = self.total_session_frames
        if total <= 0:
            return

        target_idx = (
            self.active_session_index if index is None else int(index)
        )
        if 0 <= target_idx < len(self.session_samples):
            self.session_samples.pop(target_idx)
        if 0 <= target_idx < len(self.saved_sample_ids):
            self.saved_sample_ids.pop(target_idx)

        self.session_count = max(0, len(self.saved_sample_ids), len(self.session_samples))
        if self.session_count == 0:
            self.active_session_index = -1
            self.in_session_review = False
        elif self.active_session_index >= self.session_count:
            self.active_session_index = self.session_count - 1

        if self.tracker is not None:
            self.tracker.delete_session_sample(index)

    def exit_session_review(self) -> None:
        """
        Exits session ground-truth review mode and returns to active video/webcam playback.
        """
        self.in_session_review = False
        if self.tracker is not None:
            self.tracker.exit_session_review()

    def handle_session_updated(self, payload: dict) -> None:
        """
        Synchronizes controller state with a lightweight ``session_updated`` event
        from the frontend (or full ``session_samples`` in unit tests).

        Parameters
        ----------
        payload : dict
            Event dictionary containing lightweight session metadata (``count``,
            ``active_session_index``, ``active_mode``, ``saved_sample_ids``,
            ``current_frame_index``, ``current_markers``) or ``session_samples``.
        """
        if not isinstance(payload, dict):
            return

        if isinstance(payload.get("session_samples"), list):
            self.session_samples = payload["session_samples"]
            self.session_count = len(self.session_samples)
            self.saved_sample_ids = [
                str(s.get("sample_id", ""))
                for s in self.session_samples
                if isinstance(s, dict)
            ]

        if "count" in payload and payload["count"] is not None:
            self.session_count = int(payload["count"])

        if isinstance(payload.get("saved_sample_ids"), list):
            self.saved_sample_ids = [
                str(sid) for sid in payload["saved_sample_ids"]
            ]

        if (
            "active_session_index" in payload
            and payload["active_session_index"] is not None
        ):
            self.active_session_index = int(payload["active_session_index"])

        if "active_mode" in payload:
            self.in_session_review = payload.get("active_mode") == "session"

        if (
            "current_frame_index" in payload
            and payload["current_frame_index"] is not None
        ):
            self.current_frame_index = max(
                0, int(payload["current_frame_index"])
            )
        elif (
            self.in_session_review
            and 0 <= self.active_session_index < len(self.session_samples)
        ):
            self.current_frame_index = int(
                self.session_samples[self.active_session_index].get(
                    "frame_index", self.current_frame_index
                )
            )

        if isinstance(payload.get("current_markers"), list):
            self.current_session_markers = payload["current_markers"]

    def handle_annotation_updated(self, payload: dict) -> None:
        """
        Synchronizes controller state with an ``annotation_updated`` event payload from the client.

        Parameters
        ----------
        payload : dict
            Event dictionary containing ``frame_index``, ``markers``, and optional
            ``all_frames_markers`` or ``frame_image_b64``.
        """
        if not isinstance(payload, dict):
            return

        if "frame_index" in payload and payload["frame_index"] is not None:
            self.current_frame_index = max(0, int(payload["frame_index"]))

        if (
            isinstance(payload.get("all_frames_markers"), list)
            and payload["all_frames_markers"]
        ):
            self.all_frames_markers = payload["all_frames_markers"]

        if isinstance(payload.get("markers"), list):
            if self.in_session_review:
                self.current_session_markers = payload["markers"]
                if 0 <= self.active_session_index < len(self.session_samples):
                    self.session_samples[self.active_session_index][
                        "markers"
                    ] = payload["markers"]
            self._ensure_frame_slot(self.current_frame_index)
            self.all_frames_markers[self.current_frame_index] = payload[
                "markers"
            ]

        frame_img = payload.get("frame_image_b64")
        if isinstance(frame_img, str) and frame_img:
            self.frame_images[self.current_frame_index] = frame_img
            if (
                self.in_session_review
                and 0 <= self.active_session_index < len(self.session_samples)
            ):
                self.session_samples[self.active_session_index]["image_b64"] = (
                    frame_img
                )

    def pack_session_zip(
        self, session_name: str | None = None, fps: float = 30.0
    ) -> bytes:
        """
        Compresses ``session_samples`` into a portable YOLO-Pose session ZIP archive.

        Parameters
        ----------
        session_name : str or None, optional
            Archive session name; defaults to ``self.video_name``.
        fps : float, optional
            Frame rate used for ``annotations.json`` timestamps, by default 30.0.

        Returns
        -------
        bytes
            Compressed ``.zip`` archive bytes.
        """
        return pack_session_to_zip(
            self.session_samples,
            session_name=session_name or self.video_name,
            fps=fps,
        )

    def load_session_zip(
        self, zip_bytes: bytes, replace: bool = True
    ) -> list[dict]:
        """
        Unpacks a saved session ``.zip`` archive into ``session_samples`` and pushes it
        to the frontend tracker.

        Parameters
        ----------
        zip_bytes : bytes
            Raw bytes of a saved session ``.zip`` archive.
        replace : bool, optional
            If True, replaces current session samples; if False, appends to them,
            by default True.

        Returns
        -------
        list of dict
            The active ``session_samples`` list after loading.
        """
        unpacked = unpack_session_from_zip(zip_bytes)
        if replace:
            self.session_samples = list(unpacked)
        else:
            self.session_samples.extend(unpacked)

        self.session_count = len(self.session_samples)
        self.saved_sample_ids = [
            str(s.get("sample_id", ""))
            for s in self.session_samples
            if isinstance(s, dict)
        ]

        if self.session_samples:
            self.active_session_index = 0
            self.in_session_review = True
            self.current_frame_index = int(
                self.session_samples[0].get("frame_index", 0)
            )
            self.current_session_markers = (
                self.session_samples[0].get("markers") or []
            )
        else:
            self.active_session_index = -1
            self.in_session_review = False
            self.current_session_markers = []

        if self.tracker is not None:
            self.tracker.load_session_samples(unpacked, replace=replace)

        return self.session_samples

    def export_json(self, fps: float = 30.0) -> str:
        """
        Serializes ``all_frames_markers`` into a JSON string conforming to the CV pose schema.

        Parameters
        ----------
        fps : float, optional
            Video frame rate used to compute ``timestamp_ms``, by default 30.0.

        Returns
        -------
        str
            Formatted JSON string of per-frame keypoint annotations.
        """
        export_data = format_annotations_for_export(
            self.all_frames_markers, fps=fps
        )
        return json.dumps(export_data, indent=2)

    def export_yolo_dataset_zip(self, fps: float = 30.0) -> bytes:
        """
        Packages annotated frames (raw images + Ultralytics YOLO-Pose ``.txt`` labels +
        ``data.yaml`` + ``annotations.json``) into a downloadable ZIP archive.

        Parameters
        ----------
        fps : float, optional
            Video frame rate used for ``annotations.json`` timestamps, by default 30.0.

        Returns
        -------
        bytes
            In-memory ZIP archive bytes ready for ``yolo pose train``.
        """
        if self.session_samples:
            return pack_session_to_zip(
                self.session_samples,
                session_name=self.video_name,
                fps=fps,
            )

        return build_yolo_pose_dataset_zip(
            self.all_frames_markers,
            video_path=None,
            frame_images_b64=self.frame_images,
            video_name=self.video_name,
            fps=fps,
        )


def render() -> None:
    """
    Renders the interactive Annotation Toolkit page with:
    1. Top 3-Step Workflow Bar (Model Weights, Client-Side Session Archive & Video Management, On-Demand Few-Shot Learning).
    2. Center Interactive Video & Session Ground-Truth Viewer with Thumbnail Filmstrip.
    3. Right 17-Keypoint COCO Inspector & Session Frame Action Box.
    """
    controller = AnnotationController()

    visibility_buttons: dict[int, ui.button] = {}
    edit_buttons: dict[int, ui.button] = {}
    occlude_buttons: dict[int, ui.button] = {}
    status_badges: dict[int, ui.label] = {}
    row_containers: dict[int, ui.row] = {}

    ui.label("Annotation & Few-Shot Learning Studio").classes(
        "text-3xl font-bold mb-4 text-slate-800"
    )

    def refresh_sidebar_ui() -> None:
        """Synchronizes keypoint row highlights, button states, session badges, and weights labels."""
        weights_name = Path(controller.active_weights_path).name
        is_default_w = controller.active_weights_path == DEFAULT_WEIGHTS_PATH
        weights_badge.set_text(
            f"{'Default' if is_default_w else 'Custom'}: {weights_name}"
        )

        num_gt = controller.total_session_frames
        session_count_badge.set_text(f"Session: {num_gt} GT Frames")

        if num_gt > 0:
            disp_idx = (
                controller.active_session_index + 1
                if controller.active_session_index >= 0
                else 1
            )
            nav_text = f"Saved {disp_idx}/{num_gt}"
            session_nav_badge.set_text(nav_text)
            sidebar_session_nav_badge.set_text(nav_text)
        else:
            session_nav_badge.set_text("Saved 0/0")
            sidebar_session_nav_badge.set_text("Saved 0/0")

        video_stem = Path(controller.video_name or "video").stem or "video"
        current_sample_id = (
            f"{video_stem}_frame_{controller.current_frame_index:06d}"
        )
        is_current_frame_saved = (
            controller.in_session_review
            or current_sample_id in controller.saved_sample_ids
            or any(
                s.get("sample_id") == current_sample_id
                for s in controller.session_samples
            )
        )
        if is_current_frame_saved:
            current_frame_session_status.set_text(
                "✓ Frame saved in Session (click below to update GT)"
            )
            current_frame_session_status.classes(
                replace="text-xs font-semibold text-emerald-700"
            )
        else:
            current_frame_session_status.set_text(
                "Frame not yet saved in Session"
            )
            current_frame_session_status.classes(
                replace="text-xs font-medium text-slate-500"
            )

        if controller.in_session_review and 0 <= controller.active_session_index < num_gt:
            frame_badge.set_text(
                f"GT #{controller.active_session_index + 1} [Frame {controller.current_frame_index}]"
            )
        else:
            frame_badge.set_text(f"Frame: {controller.current_frame_index}")

        if controller.active_keypoint_id is not None:
            active_meta = KEYPOINT_BY_ID.get(controller.active_keypoint_id, {})
            active_mode_badge.set_text(
                f"Editing: {controller.active_keypoint_id}. {active_meta.get('name', '')} (click or drag on canvas)"
            )
            active_mode_badge.classes(
                replace="text-xs font-semibold px-2.5 py-1 rounded bg-amber-100 text-amber-800 border border-amber-300"
            )
        else:
            active_mode_badge.set_text("Mode: View / Drag any visible marker")
            active_mode_badge.classes(
                replace="text-xs font-medium px-2.5 py-1 rounded bg-slate-100 text-slate-600 border border-slate-200"
            )

        for kp in COCO_KEYPOINTS:
            kp_id = kp["id"]
            is_vis = controller.visibility_state.get(kp_id, True)
            is_active = controller.active_keypoint_id == kp_id
            marker = controller.get_frame_marker(kp_id)

            if kp_id in visibility_buttons:
                vis_btn = visibility_buttons[kp_id]
                vis_btn.props(
                    f"icon={'visibility' if is_vis else 'visibility_off'} "
                    f"color={'primary' if is_vis else 'grey-5'}"
                )

            if kp_id in edit_buttons:
                edit_btn = edit_buttons[kp_id]
                edit_btn.props(
                    f"color={'amber-8' if is_active else 'slate-500'} "
                    f"{'unelevated' if is_active else 'flat'}"
                )

            if kp_id in row_containers:
                row_el = row_containers[kp_id]
                if is_active:
                    row_el.classes(
                        replace="w-full justify-between items-center py-1.5 border-b border-amber-200 bg-amber-50/70 px-2 rounded transition"
                    )
                else:
                    row_el.classes(
                        replace="w-full justify-between items-center py-1.5 border-b border-gray-100 last:border-0 hover:bg-slate-50 px-2 rounded transition"
                    )

            if kp_id in status_badges:
                badge = status_badges[kp_id]
                if marker is None:
                    badge.set_text("unset")
                    badge.classes(
                        replace="text-[10px] px-1.5 py-0.5 rounded bg-slate-100 text-slate-400 font-mono"
                    )
                elif float(marker.get("visibility", 1.0)) == 0.0:
                    badge.set_text("occluded")
                    badge.classes(
                        replace="text-[10px] px-1.5 py-0.5 rounded bg-amber-100 text-amber-700 font-mono"
                    )
                else:
                    badge.set_text("set")
                    badge.classes(
                        replace="text-[10px] px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-700 font-mono"
                    )

            if kp_id in occlude_buttons:
                occ_btn = occlude_buttons[kp_id]
                is_occluded = (
                    marker is not None
                    and float(marker.get("visibility", 1.0)) == 0.0
                )
                occ_btn.props(
                    f"color={'amber-7' if is_occluded else 'slate-400'}"
                )

    # --- Handlers for Top Workflow Cards ---
    async def handle_weights_upload(event) -> None:
        uploaded_name = getattr(event, "name", None) or "custom_yolo_pose.pt"
        content_obj = getattr(event, "content", None)
        if content_obj is None and hasattr(event, "file"):
            content_obj = event.file

        raw_bytes = b""
        if content_obj is not None:
            read_res = content_obj.read()
            if hasattr(read_res, "__await__"):
                raw_bytes = await read_res
            else:
                raw_bytes = read_res

        suffix = Path(uploaded_name).suffix or ".pt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_f:
            tmp_f.write(raw_bytes)
            tmp_weights_path = tmp_f.name

        controller.active_weights_path = tmp_weights_path
        refresh_sidebar_ui()
        ui.notify(
            f"Configured custom model weights: {uploaded_name} (loaded lazily on next inference/training).",
            type="positive",
        )

    def handle_use_default_weights() -> None:
        controller.active_weights_path = DEFAULT_WEIGHTS_PATH
        refresh_sidebar_ui()
        ui.notify(
            f"Restored default weights ({DEFAULT_WEIGHTS_PATH}).", type="info"
        )

    def handle_save_session_zip() -> None:
        if controller.total_session_frames == 0:
            ui.notify(
                "No saved ground-truth frames in session yet. Click 'Save Current Frame to Session (GT)' first.",
                type="warning",
            )
            return
        tracker.save_session_zip()
        ui.notify(
            f"Downloaded session archive ({controller.total_session_frames} GT frames) directly from browser.",
            type="positive",
        )

    def handle_request_train_few_shot() -> None:
        if controller.total_session_frames == 0:
            ui.notify(
                "Save at least one annotated frame to the session before running few-shot training.",
                type="warning",
            )
            return

        train_spinner.set_visibility(True)
        train_btn.disable()
        ui.notify(
            f"Packaging {controller.total_session_frames} session frames in browser and starting few-shot training...",
            type="info",
        )
        tracker.request_session_zip_for_training()

    async def handle_training_zip_ready(payload: dict) -> None:
        if not isinstance(payload, dict) or not payload.get("zip_b64"):
            train_spinner.set_visibility(False)
            train_btn.enable()
            ui.notify("Failed to package session ZIP from browser.", type="negative")
            return

        try:
            session_zip_bytes = base64.b64decode(payload["zip_b64"])
            result = await run.cpu_bound(
                run_few_shot_training,
                session_zip_bytes,
                controller.active_weights_path,
                int(controller.few_shot_epochs),
                float(controller.few_shot_lr0),
                int(controller.few_shot_freeze_layers),
            )
            controller.trained_weights_bytes = result["weights_bytes"]
            controller.trained_weights_filename = result.get(
                "weights_filename", "yolo_pose_finetuned.pt"
            )
            save_weights_btn.enable()
            use_trained_btn.enable()
            ui.notify(
                f"Few-shot training complete ({result['num_samples']} frames, {result['epochs']} epochs). "
                "Click 'Save New Weights' to download.",
                type="positive",
            )
        except Exception as exc:
            ui.notify(f"Few-shot training failed: {exc}", type="negative")
        finally:
            train_spinner.set_visibility(False)
            train_btn.enable()

    def handle_download_trained_weights() -> None:
        if not controller.trained_weights_bytes:
            ui.notify("No trained weights available yet.", type="warning")
            return
        ui.download(
            controller.trained_weights_bytes,
            filename=controller.trained_weights_filename,
        )
        ui.notify(
            f"Downloaded {controller.trained_weights_filename}.",
            type="positive",
        )

    def handle_use_trained_weights() -> None:
        if not controller.trained_weights_bytes:
            return
        with tempfile.NamedTemporaryFile(
            delete=False, suffix="_finetuned.pt"
        ) as tmp_f:
            tmp_f.write(controller.trained_weights_bytes)
            new_path = tmp_f.name
        controller.active_weights_path = new_path
        refresh_sidebar_ui()
        ui.notify(
            "Activated fine-tuned weights for subsequent inference & training.",
            type="positive",
        )

    # Hidden uploader for custom .pt weights
    weights_uploader = (
        ui.upload(
            on_upload=handle_weights_upload,
            auto_upload=True,
        )
        .props("accept=.pt,.xml")
        .classes("hidden")
    )

    # --- Top 3-Step Workflow Bar ---
    with ui.grid(columns=3).classes("w-full gap-4 mb-4"):
        # Card 1: Model Weights
        with ui.card().classes(
            "w-full p-3.5 border border-gray-200 rounded-xl shadow-sm bg-white flex flex-col justify-between"
        ):
            with ui.row().classes("w-full justify-between items-center mb-2"):
                ui.label("1. Model Weights").classes(
                    "text-sm font-bold text-slate-800"
                )
                weights_badge = ui.label(
                    f"Default: {DEFAULT_WEIGHTS_PATH}"
                ).classes(
                    "text-[11px] font-mono px-2 py-0.5 rounded bg-blue-50 text-blue-800 border border-blue-200"
                )
            with ui.row().classes("w-full items-center gap-2"):
                ui.button(
                    "Upload .pt",
                    icon="upload_file",
                    on_click=lambda: weights_uploader.run_method("pickFiles"),
                ).props("outline dense size=sm color=primary")
                ui.button(
                    "Use Default",
                    icon="restart_alt",
                    on_click=handle_use_default_weights,
                ).props("flat dense size=sm color=slate-600")

        # Card 2: Annotation Session (100% Client-Side Multi-Video GT Store)
        with ui.card().classes(
            "w-full p-3.5 border border-gray-200 rounded-xl shadow-sm bg-white flex flex-col justify-between"
        ):
            with ui.row().classes("w-full justify-between items-center mb-2"):
                ui.label("2. Annotation Session").classes(
                    "text-sm font-bold text-slate-800"
                )
                session_count_badge = ui.label("Session: 0 GT Frames").classes(
                    "text-[11px] font-mono px-2 py-0.5 rounded bg-emerald-50 text-emerald-800 border border-emerald-200"
                )
            with ui.row().classes("w-full items-center gap-2 flex-wrap"):
                ui.button(
                    "Load Session (.zip)",
                    icon="folder_open",
                    on_click=lambda: tracker.open_session_dialog(),
                ).props("outline dense size=sm color=primary")
                ui.button(
                    "Save Session (.zip)",
                    icon="save",
                    on_click=handle_save_session_zip,
                ).props("outline dense size=sm color=primary")
                ui.button(
                    "Load Video",
                    icon="video_file",
                    on_click=lambda: tracker.open_file_dialog(),
                ).props("unelevated dense size=sm color=teal-7")

        # Card 3: Few-Shot Learning
        with ui.card().classes(
            "w-full p-3.5 border border-gray-200 rounded-xl shadow-sm bg-white flex flex-col justify-between"
        ):
            with ui.row().classes("w-full justify-between items-center mb-2"):
                ui.label("3. Few-Shot Learning").classes(
                    "text-sm font-bold text-slate-800"
                )
                train_spinner = ui.spinner("dots", size="sm").classes(
                    "text-primary"
                )
                train_spinner.set_visibility(False)

            with ui.row().classes("w-full items-center gap-2 flex-wrap"):
                train_btn = ui.button(
                    "Train Few-Shot Model",
                    icon="model_training",
                    on_click=handle_request_train_few_shot,
                ).props("unelevated dense size=sm color=primary")

                save_weights_btn = ui.button(
                    "Save New Weights (.pt)",
                    icon="download",
                    on_click=handle_download_trained_weights,
                ).props("outline dense size=sm color=positive")
                save_weights_btn.disable()

                use_trained_btn = ui.button(
                    "Use Trained",
                    icon="check_circle",
                    on_click=handle_use_trained_weights,
                ).props("flat dense size=sm color=positive")
                use_trained_btn.disable()

            with ui.expansion("Advanced Settings", icon="tune").classes(
                "w-full text-xs text-slate-600 mt-1"
            ).props("dense"):
                with ui.row().classes("w-full gap-2 items-center pt-1"):
                    ui.number(
                        "Epochs",
                        value=controller.few_shot_epochs,
                        min=1,
                        max=200,
                        step=1,
                        on_change=lambda e: setattr(
                            controller,
                            "few_shot_epochs",
                            int(e.value or 10),
                        ),
                    ).props("dense outlined").classes("w-20")
                    ui.number(
                        "LR (lr0)",
                        value=controller.few_shot_lr0,
                        min=0.00001,
                        max=0.1,
                        step=0.0005,
                        format="%.5f",
                        on_change=lambda e: setattr(
                            controller,
                            "few_shot_lr0",
                            float(e.value or 0.001),
                        ),
                    ).props("dense outlined").classes("w-24")
                    ui.number(
                        "Freeze Layers",
                        value=controller.few_shot_freeze_layers,
                        min=0,
                        max=24,
                        step=1,
                        on_change=lambda e: setattr(
                            controller,
                            "few_shot_freeze_layers",
                            int(e.value if e.value is not None else 10),
                        ),
                    ).props("dense outlined").classes("w-24")

    # --- VideoTracker & Session Event Handlers ---
    async def handle_frame(payload: dict | str) -> None:
        result = await run.cpu_bound(inference.process_frame, payload)
        if (
            isinstance(result, dict)
            and "frame_index" in result
            and isinstance(result.get("markers"), list)
        ):
            f_idx = max(0, int(result["frame_index"]))
            controller._ensure_frame_slot(f_idx)
            controller.all_frames_markers[f_idx] = result["markers"]
        tracker.run_method("update_markers", result)

    def handle_upload_start(payload: dict) -> None:
        name = (
            payload.get("name", "video")
            if isinstance(payload, dict)
            else "video"
        )
        controller.video_name = Path(name).stem or "annotations"
        controller.in_session_review = False
        ui.notify(f"Uploading & analyzing {name}...", type="info")

    def handle_upload_success(payload: dict) -> None:
        if isinstance(payload, dict):
            name = payload.get("name", "video")
            controller.video_name = Path(name).stem or "annotations"
            controller.in_session_review = False
            if isinstance(payload.get("all_frames_markers"), list):
                controller.all_frames_markers = payload["all_frames_markers"]
            ui.notify(
                f"Loaded video {name} ({len(controller.all_frames_markers)} frames). "
                "Existing session GT frames remain preserved in browser.",
                type="positive",
            )
        refresh_sidebar_ui()

    def handle_annotation_event(payload: dict) -> None:
        controller.handle_annotation_updated(payload)
        refresh_sidebar_ui()

    def handle_frame_changed(payload: dict) -> None:
        if isinstance(payload, dict) and "frame_index" in payload:
            controller.current_frame_index = max(0, int(payload["frame_index"]))
            refresh_sidebar_ui()

    def handle_session_event(payload: dict) -> None:
        controller.handle_session_updated(payload)
        refresh_sidebar_ui()
        action = payload.get("action") if isinstance(payload, dict) else None
        if action == "loaded":
            ui.notify(
                f"Loaded {controller.total_session_frames} ground-truth frames into browser session.",
                type="positive",
            )

    def on_save_frame_to_session() -> None:
        tracker.save_current_frame_to_session()
        ui.notify(
            f"Saved frame #{controller.current_frame_index} to client session.",
            type="positive",
        )

    with ui.grid(columns=3).classes("w-full gap-6 h-[720px]"):
        # Left Panel (2/3 width): Unified media frame + transport & session bar
        with ui.column().classes(
            "col-span-2 h-full border border-gray-200 rounded-xl shadow-lg bg-white overflow-hidden flex flex-col justify-between"
        ):
            tracker = VideoTracker(
                on_frame=handle_frame,
                on_file_uploaded_start=handle_upload_start,
                on_file_uploaded_success=handle_upload_success,
                on_annotation_updated=handle_annotation_event,
                on_frame_changed=handle_frame_changed,
                on_session_updated=handle_session_event,
                on_training_zip_ready=handle_training_zip_ready,
                interactive=True,
            ).classes("w-full flex-grow")
            controller.bind_tracker(tracker)

            # Bottom frame transport, session capture, and GT review bar
            with ui.column().classes(
                "w-full px-4 py-2 bg-slate-50 border-t border-gray-200 gap-1.5"
            ):
                with ui.row().classes(
                    "w-full items-center justify-between flex-wrap gap-2"
                ):
                    # Video playback transport
                    with ui.row().classes("items-center gap-1.5"):
                        ui.button(
                            "Prev (-1)",
                            icon="skip_previous",
                            on_click=lambda: tracker.step_frame(-1),
                        ).props("outline dense size=sm color=primary")
                        ui.button(
                            "Play / Pause",
                            icon="play_pause",
                            on_click=tracker.toggle_playback,
                        ).props("outline dense size=sm color=primary")
                        ui.button(
                            "Next (+1)",
                            icon="skip_next",
                            on_click=lambda: tracker.step_frame(1),
                        ).props("outline dense size=sm color=primary")

                    # Save current frame to session
                    ui.button(
                        "Save Frame to Session (GT)",
                        icon="bookmark_add",
                        on_click=on_save_frame_to_session,
                    ).props("unelevated dense size=sm color=emerald-7")

                    # Session GT navigation controls
                    with ui.row().classes("items-center gap-1"):
                        ui.button(
                            icon="chevron_left",
                            on_click=lambda: tracker.step_session_sample(-1),
                        ).props("outline dense size=sm color=amber-8").tooltip(
                            "Previous Saved GT Frame"
                        )
                        session_nav_badge = ui.label("Saved 0/0").classes(
                            "text-xs font-mono font-semibold px-2 py-0.5 rounded bg-amber-50 text-amber-900 border border-amber-200"
                        )
                        ui.button(
                            icon="chevron_right",
                            on_click=lambda: tracker.step_session_sample(1),
                        ).props("outline dense size=sm color=amber-8").tooltip(
                            "Next Saved GT Frame"
                        )
                        ui.button(
                            icon="delete_sweep",
                            on_click=lambda: tracker.delete_session_sample(),
                        ).props("flat dense size=sm color=red-6").tooltip(
                            "Remove Selected GT Frame from Session"
                        )

                with ui.row().classes("w-full items-center justify-between"):
                    active_mode_badge = ui.label(
                        "Mode: View / Drag any visible marker"
                    ).classes(
                        "text-xs font-medium px-2.5 py-0.5 rounded bg-slate-100 text-slate-600 border border-slate-200"
                    )
                    frame_badge = ui.label("Frame: 0").classes(
                        "text-xs font-mono font-semibold px-2.5 py-0.5 rounded bg-blue-50 text-blue-800 border border-blue-200"
                    )

        # Right Panel (1/3 width): 17-Keypoint COCO Controls & Session Frame Action Box
        with ui.column().classes(
            "col-span-1 h-full border border-gray-200 rounded-xl shadow-lg p-5 bg-slate-50 flex flex-col"
        ):
            with ui.row().classes(
                "w-full justify-between items-center border-b pb-2 mb-2"
            ):
                ui.label("Keypoint Corrections (17)").classes(
                    "text-lg font-semibold text-slate-700"
                )
                with ui.row().classes("gap-1"):
                    def on_show_all() -> None:
                        controller.set_all_visibility(True)
                        refresh_sidebar_ui()

                    def on_hide_all() -> None:
                        controller.set_all_visibility(False)
                        refresh_sidebar_ui()

                    ui.button(
                        "Show All", icon="visibility", on_click=on_show_all
                    ).props("flat dense size=sm color=primary")
                    ui.button(
                        "Hide All", icon="visibility_off", on_click=on_hide_all
                    ).props("flat dense size=sm color=slate-500")

            ui.label(
                "Select Edit to place or reposition a keypoint on the current frame, or drag any visible point directly."
            ).classes("text-xs text-slate-600 mb-3")

            with ui.scroll_area().classes(
                "w-full flex-grow border border-gray-200 bg-white rounded p-2"
            ):
                for kp in COCO_KEYPOINTS:
                    kp_id = kp["id"]
                    kp_color = kp["color"]
                    kp_label = f"{kp_id}. {kp['name']}"

                    with ui.row().classes(
                        "w-full justify-between items-center py-1.5 border-b border-gray-100 last:border-0 hover:bg-slate-50 px-2 rounded transition"
                    ) as row_el:
                        row_containers[kp_id] = row_el

                        with ui.row().classes("items-center gap-2"):
                            ui.element("span").style(
                                f"background-color: {kp_color}; width: 12px; height: 12px; border-radius: 9999px; display: inline-block; border: 1px solid rgba(15,23,42,0.2);"
                            )
                            ui.label(kp_label).classes(
                                "text-xs font-medium text-slate-700"
                            )
                            status_badges[kp_id] = ui.label("unset").classes(
                                "text-[10px] px-1.5 py-0.5 rounded bg-slate-100 text-slate-400 font-mono"
                            )

                        with ui.row().classes("gap-1 items-center"):
                            def make_vis_handler(target_id: int):
                                def _handler() -> None:
                                    controller.toggle_keypoint_visibility(
                                        target_id
                                    )
                                    refresh_sidebar_ui()

                                return _handler

                            def make_edit_handler(target_id: int):
                                def _handler() -> None:
                                    active = controller.select_keypoint_for_edit(
                                        target_id
                                    )
                                    refresh_sidebar_ui()
                                    if active is not None:
                                        kp_name = KEYPOINT_BY_ID[active]["name"]
                                        ui.notify(
                                            f"Editing {active}. {kp_name}: click or drag on the video canvas.",
                                            type="info",
                                        )

                                return _handler

                            def make_occlude_handler(target_id: int):
                                def _handler() -> None:
                                    marker = controller.get_frame_marker(
                                        target_id
                                    )
                                    is_currently_occluded = (
                                        marker is not None
                                        and float(marker.get("visibility", 1.0))
                                        == 0.0
                                    )
                                    next_state = (
                                        "visible"
                                        if is_currently_occluded
                                        else "occluded"
                                    )
                                    controller.set_keypoint_frame_state(
                                        target_id, next_state
                                    )
                                    refresh_sidebar_ui()

                                return _handler

                            def make_delete_handler(target_id: int):
                                def _handler() -> None:
                                    controller.set_keypoint_frame_state(
                                        target_id, "delete"
                                    )
                                    refresh_sidebar_ui()

                                return _handler

                            vis_btn = (
                                ui.button(
                                    icon="visibility",
                                    on_click=make_vis_handler(kp_id),
                                )
                                .props("flat dense size=xs color=primary")
                                .tooltip("Toggle Overlay Visibility")
                            )
                            visibility_buttons[kp_id] = vis_btn

                            edit_btn = (
                                ui.button(
                                    icon="edit",
                                    on_click=make_edit_handler(kp_id),
                                )
                                .props("flat dense size=xs color=slate-500")
                                .tooltip("Select for Canvas Placement / Editing")
                            )
                            edit_buttons[kp_id] = edit_btn

                            occ_btn = (
                                ui.button(
                                    icon="lens_blur",
                                    on_click=make_occlude_handler(kp_id),
                                )
                                .props("flat dense size=xs color=slate-400")
                                .tooltip("Mark Occluded / Visible in Frame")
                            )
                            occlude_buttons[kp_id] = occ_btn

                            ui.button(
                                icon="delete_outline",
                                on_click=make_delete_handler(kp_id),
                            ).props("flat dense size=xs color=red-4").tooltip(
                                "Clear from Frame"
                            )

            # Dedicated Session Frame Action Box at bottom of right sidebar
            with ui.column().classes(
                "w-full mt-3 p-3 bg-white border border-emerald-200 rounded-lg shadow-sm gap-2"
            ):
                current_frame_session_status = ui.label(
                    "Frame not yet saved in Session"
                ).classes("text-xs font-medium text-slate-500")

                ui.button(
                    "Save Current Frame to Session (GT)",
                    icon="bookmark_add",
                    on_click=on_save_frame_to_session,
                ).classes("w-full").props("unelevated color=positive size=md")

                with ui.row().classes(
                    "w-full items-center justify-between pt-1 border-t border-gray-100"
                ):
                    with ui.row().classes("items-center gap-1"):
                        ui.button(
                            "Prev GT",
                            icon="chevron_left",
                            on_click=lambda: tracker.step_session_sample(-1),
                        ).props("outline dense size=xs color=amber-8")
                        sidebar_session_nav_badge = ui.label(
                            "Saved 0/0"
                        ).classes(
                            "text-xs font-mono font-semibold px-2 py-0.5 rounded bg-amber-50 text-amber-900 border border-amber-200"
                        )
                        ui.button(
                            "Next GT",
                            icon="chevron_right",
                            on_click=lambda: tracker.step_session_sample(1),
                        ).props("outline dense size=xs color=amber-8")

                    ui.button(
                        "Back to Video",
                        icon="videocam",
                        on_click=lambda: tracker.exit_session_review(),
                    ).props("flat dense size=xs color=primary")
