"""
Custom NiceGUI VideoTracker component bridging frontend video playback and pose tracking.
"""
from typing import Callable

from nicegui.element import Element


class VideoTracker(Element, component="video_tracker.js"):
    """
    A custom NiceGUI component for real-time webcam buffered tracking and video file upload.

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
    """

    def __init__(
        self,
        on_frame: Callable | None = None,
        on_file_uploaded_start: Callable | None = None,
        on_file_uploaded_success: Callable | None = None,
    ) -> None:
        super().__init__()

        if on_frame:
            self.on("frame_ready", lambda e: on_frame(e.args))

        if on_file_uploaded_start:
            self.on("file_uploaded_start", lambda e: on_file_uploaded_start(e.args))

        if on_file_uploaded_success:
            self.on("file_uploaded_success", lambda e: on_file_uploaded_success(e.args))

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
