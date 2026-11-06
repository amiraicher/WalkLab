"""
Custom NiceGUI VideoTracker component bridging frontend video playback, pose tracking,
and client-side ground-truth annotation session archiving.
"""
from typing import Callable

from nicegui.element import Element

VALID_KEYPOINT_FRAME_STATES = ("visible", "occluded", "delete")


class VideoTracker(Element, component="video_tracker.js"):
    """
    A custom NiceGUI component for real-time webcam buffered tracking, video file upload,
    interactive keypoint annotation, and 100% client-side session frame curation and ZIP archiving.

    Parameters
    ----------
    on_frame : Callable, optional
        Callback triggered when a dilated webcam frame is ready for inference.
        Receives a payload dictionary ``{"frame_index": int, "frame_data": str}``
        (or a base64 string).
    on_file_uploaded_start : Callable, optional
        Callback triggered when a video file upload begins.
    on_file_uploaded_success : Callable, optional
        Callback triggered when a video file upload and batch analysis succeed.
    on_annotation_updated : Callable, optional
        Callback triggered when a keypoint is placed, dragged, or has its frame state updated.
    on_frame_changed : Callable, optional
        Callback triggered when the currently displayed frame index changes.
    on_session_updated : Callable, optional
        Callback triggered when the client-side ground-truth session status changes.
    on_training_zip_ready : Callable, optional
        Callback triggered when the frontend finishes packaging the session ZIP for few-shot training.
    interactive : bool, optional
        Whether interactive keypoint editing is enabled on the overlay, by default False.
    """

    def __init__(
        self,
        on_frame: Callable | None = None,
        on_file_uploaded_start: Callable | None = None,
        on_file_uploaded_success: Callable | None = None,
        on_annotation_updated: Callable | None = None,
        on_frame_changed: Callable | None = None,
        on_session_updated: Callable | None = None,
        on_training_zip_ready: Callable | None = None,
        interactive: bool = False,
    ) -> None:
        super().__init__()
        self.interactive = bool(interactive)
        self._props["interactive"] = self.interactive

        if on_frame:
            self.on("frame_ready", lambda e: on_frame(e.args))

        if on_file_uploaded_start:
            self.on("file_uploaded_start", lambda e: on_file_uploaded_start(e.args))

        if on_file_uploaded_success:
            self.on("file_uploaded_success", lambda e: on_file_uploaded_success(e.args))

        if on_annotation_updated:
            self.on("annotation_updated", lambda e: on_annotation_updated(e.args))

        if on_frame_changed:
            self.on("frame_changed", lambda e: on_frame_changed(e.args))

        if on_session_updated:
            self.on("session_updated", lambda e: on_session_updated(e.args))

        if on_training_zip_ready:
            self.on("training_zip_ready", lambda e: on_training_zip_ready(e.args))

    def set_tracking_config(
        self,
        buffer_delay_ms: int = 200,
        dilation_factor: int = 4,
        interpolation_mode: str = "linear",
    ) -> None:
        """
        Updates the client-side real-time video buffer, frame dilation, and interpolation mode.

        Parameters
        ----------
        buffer_delay_ms : int, optional
            Display buffer delay in milliseconds, by default 200.
        dilation_factor : int, optional
            Frame dilation stride (infer every ``N``-th frame), by default 4.
        interpolation_mode : str, optional
            Interpolation strategy (``"linear"``, ``"kalman"``, or ``"none"``),
            by default ``"linear"``.
        """
        self.run_method(
            "set_tracking_config",
            {
                "bufferDelayMs": buffer_delay_ms,
                "frameDilation": dilation_factor,
                "interpolationMode": interpolation_mode,
            },
        )

    def set_keypoint_visibility(self, keypoint_id: int, visible: bool) -> None:
        """
        Toggles the overlay visibility of a specific keypoint ID on the client canvas.

        Parameters
        ----------
        keypoint_id : int
            The COCO keypoint index (0 through 16).
        visible : bool
            True to render the keypoint and its connected edges, False to hide them.
        """
        self.run_method(
            "set_keypoint_visibility",
            {"id": int(keypoint_id), "visible": bool(visible)},
        )

    def set_all_keypoints_visibility(self, visible: bool) -> None:
        """
        Sets the overlay visibility of all 17 COCO keypoints simultaneously.

        Parameters
        ----------
        visible : bool
            True to show all keypoints, False to hide all keypoints.
        """
        self.run_method("set_all_keypoints_visibility", {"visible": bool(visible)})

    def set_active_keypoint(self, keypoint_id: int | None) -> None:
        """
        Selects or deselects the active keypoint ID for interactive canvas placement and editing.

        Parameters
        ----------
        keypoint_id : int or None
            The keypoint index (0 through 16) to activate, or ``None`` to clear selection.
        """
        self.run_method(
            "set_active_keypoint",
            {"id": None if keypoint_id is None else int(keypoint_id)},
        )

    def set_frame_keypoint_state(self, keypoint_id: int, state: str) -> None:
        """
        Updates the state of a keypoint in the currently displayed video frame.

        Parameters
        ----------
        keypoint_id : int
            The keypoint index (0 through 16) to modify.
        state : str
            Target state in the current frame: ``"visible"``, ``"occluded"``, or ``"delete"``.

        Raises
        ------
        ValueError
            If ``state`` is not one of ``("visible", "occluded", "delete")``.
        """
        if state not in VALID_KEYPOINT_FRAME_STATES:
            raise ValueError(
                f"Invalid keypoint frame state '{state}'. Must be one of {VALID_KEYPOINT_FRAME_STATES}."
            )
        self.run_method(
            "set_frame_keypoint_state",
            {"id": int(keypoint_id), "state": state},
        )

    def step_frame(self, delta: int) -> None:
        """
        Steps the video playback position forward or backward by ``delta`` frames.

        Parameters
        ----------
        delta : int
            Number of frames to step (e.g., ``-1`` for previous frame, ``1`` for next frame).
        """
        self.run_method("step_frame", {"delta": int(delta)})

    def toggle_playback(self) -> None:
        """
        Toggles play/pause on the active video element in file mode.
        """
        self.run_method("toggle_playback")

    def save_current_frame_to_session(self) -> None:
        """
        Captures the currently displayed video frame and its keypoint markers into the
        client-side ground-truth annotation session.
        """
        self.run_method("save_current_frame_to_session")

    def load_session_samples(
        self, samples: list[dict], replace: bool = True
    ) -> None:
        """
        Loads a list of unpacked ground-truth session samples into the client-side session store.

        Parameters
        ----------
        samples : list of dict
            List of session sample dictionaries containing ``sample_id``, ``video_name``,
            ``frame_index``, ``image_b64``, and ``markers``.
        replace : bool, optional
            If True, replaces existing session samples; if False, appends to them,
            by default True.
        """
        self.run_method(
            "load_session_samples",
            {"samples": samples, "replace": bool(replace)},
        )

    def select_session_sample(self, index: int) -> None:
        """
        Switches the viewer into session review mode and displays the sample at ``index``.

        Parameters
        ----------
        index : int
            Zero-based index in ``sessionSamples``.
        """
        self.run_method("select_session_sample", {"index": int(index)})

    def step_session_sample(self, delta: int) -> None:
        """
        Cycles forward or backward through the saved session ground-truth frames.

        Parameters
        ----------
        delta : int
            Step offset (e.g., ``-1`` for previous sample, ``1`` for next sample).
        """
        self.run_method("step_session_sample", {"delta": int(delta)})

    def delete_session_sample(self, index: int | None = None) -> None:
        """
        Removes a saved sample from the client-side ground-truth session.

        Parameters
        ----------
        index : int or None, optional
            Zero-based sample index to delete, or ``None`` to delete the currently
            selected sample, by default None.
        """
        self.run_method(
            "delete_session_sample",
            {"index": None if index is None else int(index)},
        )

    def exit_session_review(self) -> None:
        """
        Exits static session ground-truth review mode and returns to the active video/camera stream.
        """
        self.run_method("exit_session_review")

    def open_file_dialog(self) -> None:
        """
        Programmatically opens the browser file chooser to load a new video into the session.
        """
        self.run_method("open_file_dialog")

    def save_session_zip(self) -> None:
        """
        Packages and downloads the current annotation session as a ``.zip`` archive
        directly in the browser without round-tripping through the backend.
        """
        self.run_method("save_session_zip")

    def open_session_dialog(self) -> None:
        """
        Programmatically opens the browser file chooser to load a ``.zip`` session archive
        directly in the frontend.
        """
        self.run_method("open_session_dialog")

    def request_session_zip_for_training(self) -> None:
        """
        Requests the frontend to package its in-memory session into a ``.zip`` archive
        and emit ``training_zip_ready`` to the backend for few-shot YOLO-Pose training.
        """
        self.run_method("request_session_zip_for_training")
