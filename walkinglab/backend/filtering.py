"""
Filtering and real-time frame buffer interpolation module.

Provides keypoint trajectory smoothing, NumPy-based marker interpolation across
dilated frames, and adaptive buffer delay management when single-frame inference
latency exceeds half of the configured playback buffer window.
"""
import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_BUFFER_DELAY_MS = 200
MAX_BUFFER_DELAY_MS = 2000
BUFFER_SAFETY_MULTIPLIER = 2.2
DEFAULT_DILATION_FACTOR = 4
SUPPORTED_INTERPOLATION_METHODS = ("linear", "kalman", "none")


def interpolate_frame_sequence(
    keyframes: dict[int, list[dict]],
    method: str = "linear",
) -> dict[int, list[dict]]:
    """
    Interpolates per-frame keypoint markers across frame indices using NumPy.

    For each marker ID, constructs an index vector and coordinate vectors from
    ``keyframes`` and interpolates ``x`` and ``y`` over
    ``range(index.min(), index.max() + 1)`` via ``numpy.interp``.

    Parameters
    ----------
    keyframes : dict of {int: list of dict}
        Mapping from frame index to detected keypoint marker lists. Each marker
        dictionary must contain ``"id"``, ``"x"``, and ``"y"`` keys.
    method : str, optional
        Interpolation method: ``"linear"``, ``"kalman"``, or ``"none"``.
        Defaults to ``"linear"``.

    Returns
    -------
    dict of {int: list of dict}
        Mapping from every frame index in ``[min_idx, max_idx]`` to its
        interpolated keypoint marker list.

    Raises
    ------
    ValueError
        If ``method`` is not in ``SUPPORTED_INTERPOLATION_METHODS``.
    """
    if method not in SUPPORTED_INTERPOLATION_METHODS:
        raise ValueError(
            f"Unsupported interpolation method '{method}'. "
            f"Supported methods: {SUPPORTED_INTERPOLATION_METHODS}."
        )
    if not keyframes:
        return {}

    sorted_frames = sorted(keyframes.keys())
    min_idx, max_idx = sorted_frames[0], sorted_frames[-1]
    sequence: dict[int, list[dict]] = {idx: [] for idx in range(min_idx, max_idx + 1)}

    tracks: dict[int, list[tuple[int, float, float]]] = {}
    for f_idx in sorted_frames:
        for m in keyframes[f_idx]:
            tracks.setdefault(m["id"], []).append((f_idx, float(m["x"]), float(m["y"])))

    for kp_id, points in tracks.items():
        data = np.array(points, dtype=float)
        idx_vec = data[:, 0].astype(int)
        x_vec = data[:, 1]
        y_vec = data[:, 2]
        target_idx = np.arange(idx_vec.min(), idx_vec.max() + 1)

        if method == "none":
            pos = np.searchsorted(idx_vec, target_idx, side="right") - 1
            interp_x, interp_y = x_vec[pos], y_vec[pos]
        else:
            interp_x = np.interp(target_idx, idx_vec, x_vec)
            interp_y = np.interp(target_idx, idx_vec, y_vec)

        for f_idx, x, y in zip(
            target_idx.tolist(), interp_x.tolist(), interp_y.tolist()
        ):
            sequence[f_idx].append({"id": kp_id, "x": float(x), "y": float(y)})

    return sequence


def smooth_trajectory(raw_keypoints):
    """
    Smooths a sequence of raw keypoint trajectories across frames.

    Parameters
    ----------
    raw_keypoints : list of list of dict
        Sequence of per-frame keypoint marker dictionaries.

    Returns
    -------
    None
        Placeholder stub for future trajectory smoothing implementation.
    """
    pass


class RealtimeFrameBufferInterpolator:
    """
    Manages frame-indexed keypoint interpolation and adaptive playback buffering.

    Maintains a sliding history of inferred keyframes and interpolated intermediate
    frames for real-time webcam streams. Automatically monitors single-frame
    inference time and increases ``buffer_delay_ms`` when inference exceeds half
    of the configured buffer delay.

    Parameters
    ----------
    buffer_delay_ms : int, optional
        Initial display buffer delay in milliseconds, by default ``DEFAULT_BUFFER_DELAY_MS`` (200).
    dilation_factor : int, optional
        Frame dilation stride (infer every ``N``-th frame), by default ``DEFAULT_DILATION_FACTOR`` (4).
    method : str, optional
        Interpolation method (``"linear"``, ``"kalman"``, or ``"none"``), by default ``"linear"``.
    auto_adjust_buffer : bool, optional
        Whether to automatically increase ``buffer_delay_ms`` when inference time exceeds
        half of ``buffer_delay_ms``, by default ``True``.
    max_buffer_delay_ms : int, optional
        Maximum allowed buffer delay in milliseconds when auto-adjusting,
        by default ``MAX_BUFFER_DELAY_MS`` (2000).
    max_history : int, optional
        Maximum number of frame entries retained in the internal lookup table, by default 300.
    """

    def __init__(
        self,
        buffer_delay_ms: int = DEFAULT_BUFFER_DELAY_MS,
        dilation_factor: int = DEFAULT_DILATION_FACTOR,
        method: str = "linear",
        auto_adjust_buffer: bool = True,
        max_buffer_delay_ms: int = MAX_BUFFER_DELAY_MS,
        max_history: int = 300,
    ) -> None:
        if buffer_delay_ms <= 0:
            raise ValueError("buffer_delay_ms must be positive.")
        if dilation_factor < 1:
            raise ValueError("dilation_factor must be at least 1.")
        if method not in SUPPORTED_INTERPOLATION_METHODS:
            raise ValueError(
                f"Unsupported interpolation method '{method}'. "
                f"Supported: {SUPPORTED_INTERPOLATION_METHODS}."
            )
        if max_buffer_delay_ms < buffer_delay_ms:
            raise ValueError("max_buffer_delay_ms must be >= buffer_delay_ms.")
        if max_history < 1:
            raise ValueError("max_history must be at least 1.")

        self._initial_buffer_delay_ms = int(buffer_delay_ms)
        self.buffer_delay_ms = int(buffer_delay_ms)
        self.dilation_factor = int(dilation_factor)
        self.method = method
        self.auto_adjust_buffer = bool(auto_adjust_buffer)
        self.max_buffer_delay_ms = int(max_buffer_delay_ms)
        self.max_history = int(max_history)
        self.last_buffer_adjusted: bool = False

        self._frame_markers: dict[int, list[dict]] = {}
        self._last_frame_index: int | None = None
        self._last_markers: list[dict] | None = None

    def configure(
        self,
        buffer_delay_ms: int | None = None,
        dilation_factor: int | None = None,
        method: str | None = None,
        auto_adjust_buffer: bool | None = None,
        max_buffer_delay_ms: int | None = None,
    ) -> dict:
        """
        Updates tunable buffer, dilation, and interpolation parameters at runtime.

        Parameters
        ----------
        buffer_delay_ms : int, optional
            New playback buffer delay in milliseconds (must be ``> 0``).
        dilation_factor : int, optional
            New frame dilation stride (must be ``>= 1``).
        method : str, optional
            New interpolation method (``"linear"``, ``"kalman"``, or ``"none"``).
        auto_adjust_buffer : bool, optional
            Whether to automatically resize the buffer on slow inference.
        max_buffer_delay_ms : int, optional
            Upper bound on ``buffer_delay_ms`` in milliseconds.

        Returns
        -------
        dict
            Dictionary containing the active configuration parameters.

        Raises
        ------
        ValueError
            If any supplied parameter is invalid.
        """
        if max_buffer_delay_ms is not None:
            if max_buffer_delay_ms <= 0:
                raise ValueError("max_buffer_delay_ms must be positive.")
            self.max_buffer_delay_ms = int(max_buffer_delay_ms)

        if buffer_delay_ms is not None:
            if buffer_delay_ms <= 0:
                raise ValueError("buffer_delay_ms must be positive.")
            self.buffer_delay_ms = min(int(buffer_delay_ms), self.max_buffer_delay_ms)

        if dilation_factor is not None:
            if dilation_factor < 1:
                raise ValueError("dilation_factor must be at least 1.")
            self.dilation_factor = int(dilation_factor)

        if method is not None:
            if method not in SUPPORTED_INTERPOLATION_METHODS:
                raise ValueError(
                    f"Unsupported interpolation method '{method}'. "
                    f"Supported: {SUPPORTED_INTERPOLATION_METHODS}."
                )
            self.method = method

        if auto_adjust_buffer is not None:
            self.auto_adjust_buffer = bool(auto_adjust_buffer)

        return {
            "buffer_delay_ms": self.buffer_delay_ms,
            "dilation_factor": self.dilation_factor,
            "method": self.method,
            "auto_adjust_buffer": self.auto_adjust_buffer,
            "max_buffer_delay_ms": self.max_buffer_delay_ms,
        }

    def check_and_adjust_buffer(self, inference_time_ms: float) -> bool:
        """
        Checks whether inference latency exceeds half the buffer window and adjusts if needed.

        When ``inference_time_ms > (self.buffer_delay_ms / 2.0)``, logs a warning
        and (if ``self.auto_adjust_buffer`` is enabled) increases ``self.buffer_delay_ms``
        to ``min(self.max_buffer_delay_ms, ceil(inference_time_ms * BUFFER_SAFETY_MULTIPLIER))``.

        Parameters
        ----------
        inference_time_ms : float
            Measured single-frame inference duration in milliseconds.

        Returns
        -------
        bool
            ``True`` if ``inference_time_ms`` exceeded half of ``buffer_delay_ms``,
            ``False`` otherwise.
        """
        half_buffer_ms = self.buffer_delay_ms / 2.0
        if inference_time_ms > half_buffer_ms:
            old_buffer_ms = self.buffer_delay_ms
            if self.auto_adjust_buffer:
                target_ms = int(math.ceil(inference_time_ms * BUFFER_SAFETY_MULTIPLIER))
                self.buffer_delay_ms = min(
                    self.max_buffer_delay_ms,
                    max(self.buffer_delay_ms, target_ms),
                )
            logger.warning(
                "Inference time (%.1f ms) exceeded half of the buffer delay "
                "(%.1f ms / 2 = %.1f ms). Buffer delay adjusted: %d ms -> %d ms.",
                inference_time_ms,
                float(old_buffer_ms),
                half_buffer_ms,
                old_buffer_ms,
                self.buffer_delay_ms,
            )
            self.last_buffer_adjusted = True
            return True

        self.last_buffer_adjusted = False
        return False

    def should_infer_frame(self, frame_index: int) -> bool:
        """
        Determines whether a given frame index should be dispatched for pose inference.

        Parameters
        ----------
        frame_index : int
            Zero-based sequential index of the captured video frame.

        Returns
        -------
        bool
            ``True`` if ``frame_index >= 0`` and ``frame_index % dilation_factor == 0``.
        """
        return frame_index >= 0 and (frame_index % self.dilation_factor == 0)

    def record_inference(
        self,
        frame_index: int,
        markers: list[dict],
        inference_time_ms: float | None = None,
    ) -> dict[int, list[dict]]:
        """
        Records inferred markers for ``frame_index`` and interpolates intermediate frames.

        Parameters
        ----------
        frame_index : int
            Non-negative sequential index of the inferred frame.
        markers : list of dict
            Detected normalized keypoint markers for ``frame_index``.
        inference_time_ms : float, optional
            Duration of the single-frame inference call in milliseconds. When
            provided, triggers ``check_and_adjust_buffer``.

        Returns
        -------
        dict of {int: list of dict}
            Mapping from frame index to keypoint markers for all frames spanned
            between the previous inferred frame and ``frame_index`` (inclusive).

        Raises
        ------
        ValueError
            If ``frame_index < 0``.
        """
        if frame_index < 0:
            raise ValueError(f"frame_index must be non-negative, got {frame_index}.")

        if inference_time_ms is not None:
            self.check_and_adjust_buffer(float(inference_time_ms))
        else:
            self.last_buffer_adjusted = False

        if (
            self._last_frame_index is not None
            and self._last_markers is not None
            and frame_index >= self._last_frame_index
        ):
            interpolated = interpolate_frame_sequence(
                {self._last_frame_index: self._last_markers, frame_index: markers},
                method=self.method,
            )
        else:
            interpolated = {frame_index: [dict(m) for m in markers]}

        self._frame_markers.update(interpolated)
        self._last_frame_index = frame_index
        self._last_markers = [dict(m) for m in markers]

        if len(self._frame_markers) > self.max_history:
            sorted_keys = sorted(self._frame_markers.keys())
            for old_key in sorted_keys[: len(sorted_keys) - self.max_history]:
                del self._frame_markers[old_key]

        return interpolated

    def get_markers_for_frame(self, frame_index: int) -> list[dict]:
        """
        Retrieves the interpolated or inferred marker list for a specific frame index.

        Parameters
        ----------
        frame_index : int
            Target frame index to query.

        Returns
        -------
        list of dict
            Keypoint marker list for ``frame_index``, or the nearest available
            keyframe markers if ``frame_index`` has not yet been recorded.
        """
        if frame_index in self._frame_markers:
            return [dict(m) for m in self._frame_markers[frame_index]]

        if not self._frame_markers:
            return []

        sorted_keys = sorted(self._frame_markers.keys())
        if frame_index < sorted_keys[0]:
            return [dict(m) for m in self._frame_markers[sorted_keys[0]]]
        if frame_index > sorted_keys[-1]:
            return [dict(m) for m in self._frame_markers[sorted_keys[-1]]]

        prev_idx = max(k for k in sorted_keys if k < frame_index)
        next_idx = min(k for k in sorted_keys if k > frame_index)
        seq = interpolate_frame_sequence(
            {prev_idx: self._frame_markers[prev_idx], next_idx: self._frame_markers[next_idx]},
            method=self.method,
        )
        return seq.get(frame_index, [])

    def reset(self) -> None:
        """
        Clears cached frame markers and resets buffer adjustment flags.
        """
        self._frame_markers.clear()
        self._last_frame_index = None
        self._last_markers = None
        self.last_buffer_adjusted = False
        self.buffer_delay_ms = self._initial_buffer_delay_ms
