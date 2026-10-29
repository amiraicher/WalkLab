"""
Inference module for YOLO-Pose tracking using OpenVINO native runtime.

Provides separate static-shape OpenVINO pipelines for low-latency real-time
single-frame inference and high-throughput batched video processing.
"""
import base64
import logging
import os
import sys
import time

import cv2
import numpy as np
import openvino as ov
import torch
from ultralytics import YOLO
from ultralytics.utils.ops import scale_coords

from walkinglab.backend.filtering import RealtimeFrameBufferInterpolator

REALTIME_BATCH_SIZE = 1
BATCH_SIZE = 4
MODEL_INPUT_SIZE = (640, 640)
_empirical_single_input_time = 0.41  # Fallback guess: ~2.4 FPS for B=1


def update_empirical_time(time_taken: float) -> None:
    """
    Updates the exponential moving average of single-input inference time.

    Parameters
    ----------
    time_taken : float
        Duration in seconds of the most recent single-frame inference call.
    """
    global _empirical_single_input_time
    _empirical_single_input_time = (_empirical_single_input_time * 0.8) + (time_taken * 0.2)


def get_empirical_time() -> float:
    """
    Retrieves the current exponential moving average of single-input inference time.

    Returns
    -------
    float
        Estimated single-frame inference duration in seconds.
    """
    return _empirical_single_input_time


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


class PoseEstimator:
    """
    OpenVINO-accelerated YOLO-Pose estimator with dual static-shape pipelines.

    Maintains an eagerly compiled single-frame pipeline optimized for latency
    (``B=1``) and a lazily compiled batched pipeline optimized for throughput
    (``B=4``).

    Parameters
    ----------
    model_path : str
        Path to the YOLO PyTorch (``.pt``) or OpenVINO (``.xml``) model file.
    """

    def __init__(self, model_path: str):
        """
        Initializes the native OpenVINO runtime pipelines for YOLO pose estimation.

        Parameters
        ----------
        model_path : str
            Path to the YOLO PyTorch (``.pt``) or OpenVINO (``.xml``) model file.
        """
        if model_path.endswith(".pt"):
            ov_dir = model_path.replace(".pt", "_openvino_model")
            ov_xml = os.path.join(ov_dir, model_path.replace(".pt", ".xml"))
            if not os.path.exists(ov_xml):
                logger.info(f"OpenVINO model not found. Exporting {model_path}...")
                model = YOLO(model_path)
                model.export(format="openvino")
            model_path = ov_xml

        self.img_size = MODEL_INPUT_SIZE
        self.core = ov.Core()
        self._base_model = self.core.read_model(model_path)

        self.realtime_compiled_model = self._compile_pipeline(
            batch_size=REALTIME_BATCH_SIZE,
            performance_mode=ov.properties.hint.PerformanceMode.LATENCY,
        )
        self.realtime_infer_request = self.realtime_compiled_model.create_infer_request()
        self._batch_compiled_model: ov.CompiledModel | None = None

    def _compile_pipeline(
        self,
        batch_size: int,
        performance_mode: ov.properties.hint.PerformanceMode,
    ) -> ov.CompiledModel:
        """
        Clones the base OpenVINO model, reshapes to a static batch size, and compiles it.

        Parameters
        ----------
        batch_size : int
            Static batch dimension to set on input port 0.
        performance_mode : openvino.properties.hint.PerformanceMode
            OpenVINO performance hint (e.g., ``LATENCY`` or ``THROUGHPUT``).

        Returns
        -------
        openvino.CompiledModel
            The compiled OpenVINO model bound to the CPU device.
        """
        model_clone = self._base_model.clone()
        model_clone.reshape(
            {0: ov.PartialShape([batch_size, 3, self.img_size[0], self.img_size[1]])}
        )
        return self.core.compile_model(
            model_clone,
            "CPU",
            {ov.properties.hint.performance_mode(): performance_mode},
        )

    @property
    def batch_compiled_model(self) -> ov.CompiledModel:
        """
        Returns the throughput-optimized batch compiled model, compiling it lazily on first access.

        Returns
        -------
        openvino.CompiledModel
            The compiled OpenVINO model configured for static ``BATCH_SIZE`` throughput.
        """
        if self._batch_compiled_model is None:
            self._batch_compiled_model = self._compile_pipeline(
                batch_size=BATCH_SIZE,
                performance_mode=ov.properties.hint.PerformanceMode.THROUGHPUT,
            )
        return self._batch_compiled_model

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, tuple]:
        """
        Prepares a BGR image for YOLO-Pose inference using letterbox resizing.

        Parameters
        ----------
        image : numpy.ndarray
            The original BGR image loaded via OpenCV with shape ``(H, W, C)``.

        Returns
        -------
        tuple of (numpy.ndarray, float, tuple of float)
            A tuple ``(preprocessed_image_tensor, scale_ratio, padding_tuple)``
            where ``preprocessed_image_tensor`` has shape ``(1, 3, 640, 640)``.
        """
        height, width = image.shape[:2]
        scale_ratio = min(self.img_size[0] / height, self.img_size[1] / width)
        unpadded_dimensions = (
            int(round(width * scale_ratio)),
            int(round(height * scale_ratio)),
        )
        pad_w = (self.img_size[1] - unpadded_dimensions[0]) / 2
        pad_h = (self.img_size[0] - unpadded_dimensions[1]) / 2

        if (width, height) != unpadded_dimensions:
            img = cv2.resize(image, unpadded_dimensions, interpolation=cv2.INTER_LINEAR)
        else:
            img = image.copy()

        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        # Convert BGR to RGB, HWC to CHW
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        # Normalize and add batch dimension
        img_tensor = img.astype(np.float32) / 255.0
        img_tensor = np.expand_dims(img_tensor, axis=0)

        return img_tensor, scale_ratio, (pad_w, pad_h)

    def postprocess(self, output: np.ndarray, orig_img_shape: tuple) -> list[dict]:
        """
        Extracts end-to-end predicted keypoints and maps coordinates back to original dimensions.

        Parameters
        ----------
        output : numpy.ndarray
            The raw output tensor from the OpenVINO model with shape ``(1, 300, 57)``.
        orig_img_shape : tuple
            The shape of the original image ``(H, W, C)``.

        Returns
        -------
        list of dict
            A list of visible keypoint dictionaries ``{"id": int, "x": float, "y": float}``
            normalized to ``[0, 1]`` relative to the original image dimensions.
        """
        # output is [1, 300, 57] where 57 = 4(bbox) + 1(conf) + 1(cls) + 51(17 keypoints * 3)
        preds = output[0]  # shape [300, 57]

        # Filter by confidence > 0.5
        confs = preds[:, 4]
        valid_mask = confs > 0.5
        valid_preds = preds[valid_mask]

        if len(valid_preds) == 0:
            return []

        # Sort by confidence descending
        valid_preds = valid_preds[valid_preds[:, 4].argsort()[::-1]]

        # Take the best detection (main person)
        best_det = valid_preds[0]

        # Extract keypoints: index 6 onwards
        # shape [17*3] -> [1, 17, 3] to match scale_coords expected input
        kpts = best_det[6:].reshape(1, 17, 3)

        scaled_kpts = scale_coords(self.img_size, torch.from_numpy(kpts), orig_img_shape)

        markers = []
        height, width = orig_img_shape[:2]
        for kp_idx, keypoint in enumerate(scaled_kpts[0]):
            x, y, visibility = keypoint.tolist()
            if visibility > 0.5:
                markers.append({"id": kp_idx, "x": x / width, "y": y / height})

        return markers

    def infer(self, image: np.ndarray) -> list[dict]:
        """
        Runs the full synchronous single-frame inference pipeline on a BGR image array.

        Parameters
        ----------
        image : numpy.ndarray
            The input BGR image array with shape ``(H, W, C)``.

        Returns
        -------
        list of dict
            List of normalized keypoint marker coordinates ``{"x": float, "y": float}``.
        """
        t_start = time.perf_counter()

        img_tensor, _, _ = self.preprocess(image)
        t_pre = time.perf_counter()

        results = self.realtime_infer_request.infer([img_tensor])
        output = next(iter(results.values()))
        t_inf = time.perf_counter()

        markers = self.postprocess(output, image.shape)
        t_end = time.perf_counter()

        update_empirical_time(t_end - t_start)

        logger.debug(
            f"Timing - Preprocess: {(t_pre - t_start) * 1000:.1f}ms, "
            f"Inference: {(t_inf - t_pre) * 1000:.1f}ms, "
            f"Postprocess: {(t_end - t_inf) * 1000:.1f}ms, "
            f"Total: {(t_end - t_start) * 1000:.1f}ms"
        )

        return markers

    def predict(self, frame_data_b64: str) -> list[dict]:
        """
        Decodes a base64-encoded image frame and delegates to ``infer``.

        Parameters
        ----------
        frame_data_b64 : str
            Base64-encoded image string, optionally prefixed with a data URI header
            (e.g., ``data:image/jpeg;base64,...``).

        Returns
        -------
        list of dict
            List of normalized keypoint marker coordinates ``{"x": float, "y": float}``.
        """
        encoded = frame_data_b64.split(",", 1)[1] if "," in frame_data_b64 else frame_data_b64
        data = base64.b64decode(encoded)
        np_arr = np.frombuffer(data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return self.infer(image)


def _pad_batch(
    batch_tensors: list[np.ndarray],
    target_batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """
    Stacks a list of single-frame CHW tensors into a static-sized batch with zero-padding.

    Parameters
    ----------
    batch_tensors : list of numpy.ndarray
        Non-empty list of preprocessed image tensors, each of shape ``(3, H, W)``,
        containing at most ``target_batch_size`` elements.
    target_batch_size : int, optional
        The required static batch size along axis 0, by default ``BATCH_SIZE``.

    Returns
    -------
    numpy.ndarray
        A float32 batch tensor of exact shape ``(target_batch_size, 3, H, W)``.
    """
    batched = np.stack(batch_tensors, axis=0).astype(np.float32, copy=False)
    num_frames = len(batch_tensors)
    if num_frames < target_batch_size:
        pad_count = target_batch_size - num_frames
        pad_width = ((0, pad_count), (0, 0), (0, 0), (0, 0))
        batched = np.pad(batched, pad_width, mode="constant", constant_values=0.0)
    return batched


# Global singletons for the estimator and real-time frame buffer interpolator
estimator: PoseEstimator | None = None
realtime_interpolator: RealtimeFrameBufferInterpolator | None = None


def get_estimator() -> PoseEstimator:
    """
    Retrieves the global ``PoseEstimator`` singleton, initializing it if necessary.

    Returns
    -------
    PoseEstimator
        The initialized pose estimator instance.
    """
    global estimator
    if estimator is None:
        estimator = PoseEstimator("yolo26s-pose.pt")
    return estimator


def get_realtime_interpolator() -> RealtimeFrameBufferInterpolator:
    """
    Retrieves the global ``RealtimeFrameBufferInterpolator`` singleton.

    Returns
    -------
    RealtimeFrameBufferInterpolator
        The active real-time frame buffer interpolator instance.
    """
    global realtime_interpolator
    if realtime_interpolator is None:
        realtime_interpolator = RealtimeFrameBufferInterpolator()
    return realtime_interpolator


def configure_realtime_interpolator(
    buffer_delay_ms: int | None = None,
    dilation_factor: int | None = None,
    method: str | None = None,
    auto_adjust_buffer: bool | None = None,
    max_buffer_delay_ms: int | None = None,
) -> dict:
    """
    Configures the global ``RealtimeFrameBufferInterpolator`` parameters at runtime.

    Parameters
    ----------
    buffer_delay_ms : int, optional
        Target playback buffer delay in milliseconds.
    dilation_factor : int, optional
        Frame dilation stride (infer every ``N``-th frame).
    method : str, optional
        Marker interpolation strategy (``"linear"``, ``"kalman"``, or ``"none"``).
    auto_adjust_buffer : bool, optional
        Whether to automatically increase ``buffer_delay_ms`` when inference
        exceeds half of the current buffer delay.
    max_buffer_delay_ms : int, optional
        Maximum allowed buffer delay in milliseconds.

    Returns
    -------
    dict
        Active configuration dictionary from the interpolator.
    """
    interpolator = get_realtime_interpolator()
    return interpolator.configure(
        buffer_delay_ms=buffer_delay_ms,
        dilation_factor=dilation_factor,
        method=method,
        auto_adjust_buffer=auto_adjust_buffer,
        max_buffer_delay_ms=max_buffer_delay_ms,
    )


def process_frame(
    frame_data: str | dict,
    frame_index: int | None = None,
) -> list[dict] | dict:
    """
    Processes a real-time webcam frame and returns marker coordinates or an indexed payload.

    Supports two invocation modes:
    1. **Legacy mode**: When ``frame_data`` is a base64 ``str`` and ``frame_index`` is
       ``None``, returns ``list[dict]`` of normalized keypoint markers directly.
    2. **Indexed real-time mode**: When ``frame_data`` is a ``dict`` containing
       ``"frame_index"`` and ``"frame_data"`` (or when ``frame_index`` is explicitly
       provided), runs single-frame inference, records the result in the real-time
       interpolator, interpolates intermediate dilated frames, and checks whether
       inference latency exceeded half of the buffer window.

    Parameters
    ----------
    frame_data : str or dict
        Either a base64-encoded image string, or a dictionary with keys
        ``"frame_index"`` (int) and ``"frame_data"`` (str), plus optional tuning
        keys ``"buffer_delay_ms"``, ``"dilation_factor"``, ``"interpolation_method"``,
        and ``"inference_time_ms"``.
    frame_index : int, optional
        Explicit sequential index of the frame when ``frame_data`` is passed as a ``str``.

    Returns
    -------
    list of dict or dict
        In legacy mode, a list of keypoint marker dictionaries. In indexed mode,
        a dictionary containing ``frame_index``, ``markers``, ``interpolated_frames``,
        ``interpolation_method``, ``buffer_delay_ms``, ``buffer_adjusted``,
        ``inference_time_ms``, and ``dilation_factor``.
    """
    if isinstance(frame_data, str) and frame_index is None:
        return get_estimator().predict(frame_data)

    interpolator = get_realtime_interpolator()

    if isinstance(frame_data, dict):
        idx = int(frame_data.get("frame_index", 0 if frame_index is None else frame_index))
        b64_str = frame_data["frame_data"]
        if any(
            key in frame_data
            for key in ("buffer_delay_ms", "dilation_factor", "interpolation_method")
        ):
            interpolator.configure(
                buffer_delay_ms=frame_data.get("buffer_delay_ms"),
                dilation_factor=frame_data.get("dilation_factor"),
                method=frame_data.get("interpolation_method"),
            )
    else:
        idx = int(frame_index)  # type: ignore[arg-type]
        b64_str = frame_data

    t_start = time.perf_counter()
    markers = get_estimator().predict(b64_str)
    measured_ms = (time.perf_counter() - t_start) * 1000.0

    if isinstance(frame_data, dict) and "inference_time_ms" in frame_data:
        inference_time_ms = float(frame_data["inference_time_ms"])
    else:
        inference_time_ms = measured_ms

    interpolated = interpolator.record_inference(
        idx,
        markers,
        inference_time_ms=inference_time_ms,
    )

    return {
        "frame_index": idx,
        "markers": markers,
        "interpolated_frames": interpolated,
        "interpolation_method": interpolator.method,
        "buffer_delay_ms": interpolator.buffer_delay_ms,
        "buffer_adjusted": interpolator.last_buffer_adjusted,
        "inference_time_ms": inference_time_ms,
        "dilation_factor": interpolator.dilation_factor,
    }


def process_video_file(file_info: dict) -> list:
    """
    Processes a video file asynchronously using OpenVINO ``AsyncInferQueue`` with static batching.

    Parameters
    ----------
    file_info : dict
        Dictionary containing video metadata, including the ``"path"`` key pointing
        to the video file on disk.

    Returns
    -------
    list of list of dict
        Ordered list of keypoint marker lists corresponding to each decoded frame.
    """
    video_path = file_info.get("path")
    if not video_path:
        logger.error("No video path provided.")
        return []

    est = get_estimator()
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Pre-allocate array for all markers
    all_markers = [None] * total_frames

    # 0 = auto-detect optimal concurrent requests based on CPU cores
    infer_queue = ov.AsyncInferQueue(est.batch_compiled_model, 0)

    def callback(request, userdata):
        batch_indices, batch_shapes = userdata
        output = next(iter(request.results.values()))
        # output is shape [BATCH_SIZE, 300, 57]; only unpack valid frames
        for i, (idx, orig_shape) in enumerate(zip(batch_indices, batch_shapes)):
            # postprocess expects shape [1, 300, 57]
            single_output = np.expand_dims(output[i], axis=0)
            markers = est.postprocess(single_output, orig_shape)
            all_markers[idx] = markers

    infer_queue.set_callback(callback)

    frame_idx = 0
    t_start = time.perf_counter()
    logger.info(
        f"🚀 [Batch Mode] Starting Async inference (B={BATCH_SIZE}) on {video_path}..."
    )

    batch_tensors = []
    batch_shapes = []
    batch_indices = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_tensor, _, _ = est.preprocess(frame)
        # img_tensor is [1, 3, 640, 640], extract to [3, 640, 640] for stacking
        batch_tensors.append(img_tensor[0])
        batch_shapes.append(frame.shape)
        batch_indices.append(frame_idx)

        frame_idx += 1

        # Dispatch when batch is full
        if len(batch_tensors) == BATCH_SIZE:
            batched_input = _pad_batch(batch_tensors, BATCH_SIZE)
            infer_queue.start_async(
                [batched_input], userdata=(batch_indices, batch_shapes)
            )
            batch_tensors = []
            batch_shapes = []
            batch_indices = []

        if frame_idx % 80 == 0:
            logger.info(f"[Batch Mode] Queued {frame_idx} frames...")

    # Dispatch remaining frames with zero-padding to match static BATCH_SIZE
    if batch_tensors:
        batched_input = _pad_batch(batch_tensors, BATCH_SIZE)
        infer_queue.start_async(
            [batched_input], userdata=(batch_indices, batch_shapes)
        )

    # Wait for the remaining frames in the queue to finish processing
    infer_queue.wait_all()
    cap.release()

    t_end = time.perf_counter()
    if frame_idx > 0:
        fps = frame_idx / (t_end - t_start)
        logger.info(
            f"✅ [Batch Mode] Complete! Processed {frame_idx} frames in "
            f"{t_end - t_start:.2f}s ({fps:.1f} FPS)"
        )

    # Return only the frames we actually read
    return all_markers[:frame_idx]

