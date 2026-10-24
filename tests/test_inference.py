"""
Automated test suite for the split real-time and batch OpenVINO inference pipelines.
"""
import base64
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

from walkinglab.backend.inference import (
    BATCH_SIZE,
    _pad_batch,
    get_estimator,
    process_frame,
    process_video_file,
)


def test_realtime_pipeline_compilation():
    """
    Verifies that the real-time OpenVINO pipeline is compiled with a static
    shape of [1, 3, 640, 640] and LATENCY performance mode, and that legacy
    attributes are absent.
    """
    est = get_estimator()
    assert est is not None
    assert est.realtime_compiled_model is not None
    assert est.realtime_infer_request is not None

    assert not hasattr(est, "compiled_model")
    assert not hasattr(est, "infer_request")

    input_port = est.realtime_compiled_model.input(0)
    assert not input_port.partial_shape.is_dynamic
    assert input_port.shape == ov.Shape([1, 3, 640, 640])

    perf_mode = est.realtime_compiled_model.get_property(
        ov.properties.hint.performance_mode()
    )
    assert perf_mode == ov.properties.hint.PerformanceMode.LATENCY


def test_batch_pipeline_compilation():
    """
    Verifies that the batch OpenVINO pipeline lazily compiles with a static
    shape of [BATCH_SIZE, 3, 640, 640] and THROUGHPUT performance mode.
    """
    est = get_estimator()
    batch_model = est.batch_compiled_model
    assert batch_model is not None

    input_port = batch_model.input(0)
    assert not input_port.partial_shape.is_dynamic
    assert input_port.shape == ov.Shape([BATCH_SIZE, 3, 640, 640])

    perf_mode = batch_model.get_property(
        ov.properties.hint.performance_mode()
    )
    assert perf_mode == ov.properties.hint.PerformanceMode.THROUGHPUT


def test_realtime_infer_and_predict():
    """
    Verifies that real-time inference via both `infer` (numpy array) and
    `predict` / `process_frame` (base64 JPEG string) returns a list.
    """
    est = get_estimator()
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)

    markers_infer = est.infer(dummy_image)
    assert isinstance(markers_infer, list)

    success, encoded_img = cv2.imencode(".jpg", dummy_image)
    assert success
    b64_payload = base64.b64encode(encoded_img.tobytes()).decode("utf-8")
    b64_str = f"data:image/jpeg;base64,{b64_payload}"

    markers_predict = est.predict(b64_str)
    assert isinstance(markers_predict, list)

    markers_frame = process_frame(b64_str)
    assert isinstance(markers_frame, list)


def test_pad_batch_static_shape():
    """
    Verifies that `_pad_batch` pads partial and full batches of (3, 640, 640)
    tensors to exact static shape (BATCH_SIZE, 3, 640, 640) while preserving
    original tensor values.
    """
    for count in range(1, BATCH_SIZE + 1):
        tensors = [
            np.full((3, 640, 640), fill_value=float(idx + 1), dtype=np.float32)
            for idx in range(count)
        ]
        batched = _pad_batch(tensors, BATCH_SIZE)

        assert isinstance(batched, np.ndarray)
        assert batched.shape == (BATCH_SIZE, 3, 640, 640)

        for idx in range(count):
            assert np.allclose(batched[idx], float(idx + 1))

        if count < BATCH_SIZE:
            assert np.allclose(batched[count:], 0.0)


def test_process_video_file_with_partial_batch(tmp_path: Path):
    """
    Verifies that `process_video_file` processes a 6-frame synthetic video
    (1 full batch of 4 + 1 partial batch of 2) without failing on static-shape
    padding and returns markers for all 6 frames.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory fixture.
    """
    video_path = tmp_path / "synthetic_walk.mp4"
    width, height = 640, 480
    num_frames = 6

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (width, height))
    assert writer.isOpened()

    for idx in range(num_frames):
        frame = np.full((height, width, 3), fill_value=idx * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    results = process_video_file({"path": str(video_path)})

    assert isinstance(results, list)
    assert len(results) == num_frames
    for frame_markers in results:
        assert isinstance(frame_markers, list)
