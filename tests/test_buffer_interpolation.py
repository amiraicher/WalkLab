"""
Automated test suite for the real-time frame delay buffer, frame dilation,
linear/tunable marker interpolation, and adaptive latency buffer expansion.
"""
import base64
import logging
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from walkinglab.backend.filtering import (
    DEFAULT_BUFFER_DELAY_MS,
    DEFAULT_DILATION_FACTOR,
    SUPPORTED_INTERPOLATION_METHODS,
    RealtimeFrameBufferInterpolator,
    interpolate_frame_sequence,
)
from walkinglab.backend.inference import (
    configure_realtime_interpolator,
    get_realtime_interpolator,
    process_frame,
)
from walkinglab.ui.components.video_tracker import VideoTracker


def _create_dummy_b64_frame(width: int = 320, height: int = 240) -> str:
    """
    Creates a synthetic blank JPEG image encoded as a base64 data URI string.

    Parameters
    ----------
    width : int, optional
        Width of the synthetic image in pixels, by default 320.
    height : int, optional
        Height of the synthetic image in pixels, by default 240.

    Returns
    -------
    str
        Base64-encoded JPEG data URI string.
    """
    dummy_image = np.zeros((height, width, 3), dtype=np.uint8)
    success, encoded_img = cv2.imencode(".jpg", dummy_image)
    assert success
    b64_payload = base64.b64encode(encoded_img.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_payload}"


def test_interpolate_frame_sequence_id_matching():
    """
    Verifies that `interpolate_frame_sequence` interpolates keypoints by `"id"`
    across keyframes even when marker ordering differs between frames.
    """
    markers_start = [
        {"id": 2, "x": 0.20, "y": 0.20},
        {"id": 1, "x": 0.10, "y": 0.10},
    ]
    markers_end = [
        {"id": 1, "x": 0.30, "y": 0.50},
        {"id": 2, "x": 0.60, "y": 0.40},
    ]

    seq = interpolate_frame_sequence({0: markers_start, 2: markers_end}, method="linear")
    by_id = {marker["id"]: marker for marker in seq[1]}

    assert set(by_id.keys()) == {1, 2}
    assert np.isclose(by_id[1]["x"], 0.20)
    assert np.isclose(by_id[1]["y"], 0.30)
    assert np.isclose(by_id[2]["x"], 0.40)
    assert np.isclose(by_id[2]["y"], 0.30)


def test_interpolate_frame_sequence_methods_and_validation():
    """
    Verifies `interpolate_frame_sequence` across a 4-frame gap (frames 0..4,
    simulating 40 FPS video vs 10 FPS inference) for `"linear"`, `"kalman"`,
    and `"none"` methods using `{idx: markers}` dictionary inputs, as well as
    `ValueError` validation for unsupported methods and empty keyframe dicts.

    Raises
    ------
    AssertionError
        If frame indices, interpolated coordinates, or validation errors do
        not match specification.
    """
    markers_0 = [{"id": 0, "x": 0.0, "y": 0.0}]
    markers_4 = [{"id": 0, "x": 0.4, "y": 0.8}]

    seq_linear = interpolate_frame_sequence(
        {0: markers_0, 4: markers_4}, method="linear"
    )
    assert set(seq_linear.keys()) == {0, 1, 2, 3, 4}
    for idx in range(5):
        assert np.isclose(seq_linear[idx][0]["x"], idx * 0.1)
        assert np.isclose(seq_linear[idx][0]["y"], idx * 0.2)

    seq_kalman = interpolate_frame_sequence(
        {0: markers_0, 4: markers_4}, method="kalman"
    )
    assert set(seq_kalman.keys()) == {0, 1, 2, 3, 4}
    assert np.isclose(seq_kalman[2][0]["x"], 0.2)
    assert np.isclose(seq_kalman[2][0]["y"], 0.4)

    seq_none = interpolate_frame_sequence(
        {0: markers_0, 4: markers_4}, method="none"
    )
    assert set(seq_none.keys()) == {0, 1, 2, 3, 4}
    for idx in range(4):
        assert np.isclose(seq_none[idx][0]["x"], 0.0)
        assert np.isclose(seq_none[idx][0]["y"], 0.0)
    assert np.isclose(seq_none[4][0]["x"], 0.4)
    assert np.isclose(seq_none[4][0]["y"], 0.8)

    assert interpolate_frame_sequence({}, method="linear") == {}

    with pytest.raises(ValueError, match="Unsupported"):
        interpolate_frame_sequence({0: markers_0, 4: markers_4}, method="cubic_spline")


def test_interpolate_frame_sequence_numpy_dict_and_multi_marker_ids():
    """
    Verifies the simplified NumPy-based `interpolate_frame_sequence` with
    `{idx: markers}` dictionary input across multiple keyframes and marker IDs,
    ensuring exact `np.interp` behavior across `range(index.min(), index.max() + 1)`
    per marker `id`.
    """
    keyframe_dict = {
        2: [
            {"id": 10, "x": 0.10, "y": 0.20},
            {"id": 20, "x": 0.80, "y": 0.90},
        ],
        6: [
            {"id": 10, "x": 0.50, "y": 0.60},
            {"id": 20, "x": 0.40, "y": 0.50},
        ],
        10: [
            {"id": 10, "x": 0.90, "y": 0.20},
            {"id": 20, "x": 0.20, "y": 0.10},
        ],
    }

    known_indices = np.array(sorted(keyframe_dict.keys()), dtype=float)
    target_indices = np.arange(int(known_indices.min()), int(known_indices.max()) + 1)

    seq_from_dict = interpolate_frame_sequence(keyframe_dict, method="linear")
    assert list(seq_from_dict.keys()) == list(range(2, 11))

    for marker_id, known_x, known_y in [
        (10, [0.10, 0.50, 0.90], [0.20, 0.60, 0.20]),
        (20, [0.80, 0.40, 0.20], [0.90, 0.50, 0.10]),
    ]:
        expected_x = np.interp(target_indices, known_indices, known_x)
        expected_y = np.interp(target_indices, known_indices, known_y)

        for offset, frame_idx in enumerate(target_indices):
            frame_by_id = {m["id"]: m for m in seq_from_dict[int(frame_idx)]}
            assert marker_id in frame_by_id
            assert np.isclose(frame_by_id[marker_id]["x"], expected_x[offset])
            assert np.isclose(frame_by_id[marker_id]["y"], expected_y[offset])



def test_realtime_frame_buffer_interpolator_dilation_and_tuning():
    """
    Verifies default parameters, 1-in-4 frame dilation (`should_infer_frame`),
    intermediate frame population (`record_inference` and `get_markers_for_frame`),
    and runtime tunability (`configure`).
    """
    manager = RealtimeFrameBufferInterpolator()
    assert manager.buffer_delay_ms == DEFAULT_BUFFER_DELAY_MS == 200
    assert manager.dilation_factor == DEFAULT_DILATION_FACTOR == 4
    assert manager.method == "linear"
    assert "linear" in SUPPORTED_INTERPOLATION_METHODS
    assert manager.auto_adjust_buffer is True

    # Frame dilation: 1 in 4 frames (0, 4, 8, 12)
    inferred_indices = [idx for idx in range(13) if manager.should_infer_frame(idx)]
    assert inferred_indices == [0, 4, 8, 12]

    markers_0 = [{"id": 0, "x": 0.1, "y": 0.2}]
    markers_4 = [{"id": 0, "x": 0.5, "y": 0.6}]

    populated_0 = manager.record_inference(0, markers_0, inference_time_ms=45.0)
    assert set(populated_0.keys()) == {0}

    populated_4 = manager.record_inference(4, markers_4, inference_time_ms=50.0)
    assert set(populated_4.keys()) == {0, 1, 2, 3, 4}

    for frame_offset in range(5):
        frame_markers = manager.get_markers_for_frame(frame_offset)
        assert frame_markers is not None
        assert np.isclose(frame_markers[0]["x"], 0.1 + 0.1 * frame_offset)
        assert np.isclose(frame_markers[0]["y"], 0.2 + 0.1 * frame_offset)

    # Verify runtime reconfiguration
    manager.configure(buffer_delay_ms=250, dilation_factor=5, method="kalman")
    assert manager.buffer_delay_ms == 250
    assert manager.dilation_factor == 5
    assert manager.method == "kalman"
    assert manager.should_infer_frame(5) is True
    assert manager.should_infer_frame(4) is False


def test_adaptive_buffer_expansion_and_warning_log(
    caplog: pytest.LogCaptureFixture,
):
    """
    Verifies that when single-frame inference latency exceeds half of the
    configured buffer delay (`inference_time_ms > buffer_delay_ms / 2.0`),
    a `WARNING` log is emitted and `buffer_delay_ms` is automatically expanded.

    Parameters
    ----------
    caplog : pytest.LogCaptureFixture
        Pytest fixture for capturing log messages.
    """
    manager = RealtimeFrameBufferInterpolator(
        buffer_delay_ms=200,
        dilation_factor=4,
        method="linear",
        auto_adjust_buffer=True,
    )

    with caplog.at_level(logging.WARNING, logger="walkinglab.backend.filtering"):
        # 80.0 ms <= 200 / 2.0 (100.0 ms): no warning or expansion
        adjusted_low = manager.check_and_adjust_buffer(80.0)
        assert adjusted_low is False
        assert manager.last_buffer_adjusted is False
        assert manager.buffer_delay_ms == 200
        assert len(caplog.records) == 0

        # 150.0 ms > 200 / 2.0 (100.0 ms): warning emitted and buffer expanded
        markers_0 = [{"id": 0, "x": 0.2, "y": 0.3}]
        manager.record_inference(0, markers_0, inference_time_ms=150.0)

        assert manager.last_buffer_adjusted is True
        expected_expanded_ms = int(math.ceil(150.0 * 2.2))
        assert manager.buffer_delay_ms >= 300
        assert manager.buffer_delay_ms == expected_expanded_ms == 330
        assert len(caplog.records) >= 1
        assert any(
            record.levelname == "WARNING" and "150.0" in record.message
            for record in caplog.records
        )


def test_process_frame_indexed_and_backward_compatible(
    caplog: pytest.LogCaptureFixture,
):
    """
    Verifies that `inference.process_frame` supports indexed frame dictionaries
    with frame association, linear interpolation, and high-latency buffer
    adaptation while maintaining backward compatibility with raw base64 strings.

    Parameters
    ----------
    caplog : pytest.LogCaptureFixture
        Pytest fixture for capturing log messages.
    """
    interpolator = get_realtime_interpolator()
    interpolator.reset()
    configure_realtime_interpolator(
        buffer_delay_ms=200,
        dilation_factor=4,
        method="linear",
        auto_adjust_buffer=True,
    )

    b64_str = _create_dummy_b64_frame()

    # Indexed frame 0
    res_0 = process_frame(
        {
            "frame_index": 0,
            "frame_data": b64_str,
            "inference_time_ms": 40.0,
        }
    )
    assert isinstance(res_0, dict)
    assert res_0["frame_index"] == 0
    assert isinstance(res_0["markers"], list)
    assert 0 in res_0["interpolated_frames"]
    assert res_0["buffer_delay_ms"] == 200
    assert res_0["buffer_adjusted"] is False
    assert res_0["dilation_factor"] == 4
    assert isinstance(res_0["inference_time_ms"], float)

    # Indexed frame 4: should return interpolated frames 0..4
    res_4 = process_frame(
        {
            "frame_index": 4,
            "frame_data": b64_str,
            "inference_time_ms": 45.0,
        }
    )
    assert isinstance(res_4, dict)
    assert res_4["frame_index"] == 4
    assert set(res_4["interpolated_frames"].keys()) == {0, 1, 2, 3, 4}
    assert res_4["buffer_adjusted"] is False
    assert res_4["buffer_delay_ms"] == 200

    # Simulated high latency (250.0 ms > 100.0 ms threshold) on frame 8
    with caplog.at_level(logging.WARNING, logger="walkinglab.backend.filtering"):
        res_8 = process_frame(
            {
                "frame_index": 8,
                "frame_data": b64_str,
                "inference_time_ms": 250.0,
            }
        )
        assert res_8["frame_index"] == 8
        assert res_8["buffer_adjusted"] is True
        assert res_8["buffer_delay_ms"] >= 500
        assert any(record.levelname == "WARNING" for record in caplog.records)

    # Backward compatibility: raw base64 string returns list
    legacy_result = process_frame(b64_str)
    assert isinstance(legacy_result, list)


def test_frontend_video_tracker_component_structure():
    """
    Verifies that `video_tracker.js` and `VideoTracker` implement the 200ms
    delayed canvas buffer, 1-in-4 frame dilation, linear marker interpolation,
    indexed frame payloads, and dynamic buffer latency warnings.
    """
    js_path = (
        Path(__file__).resolve().parent.parent
        / "walkinglab"
        / "ui"
        / "components"
        / "video_tracker.js"
    )
    assert js_path.exists()
    js_content = js_path.read_text(encoding="utf-8")

    required_tokens = [
        "bufferDelayMs: 200",
        "frameDilation: 4",
        "interpolationMode: 'linear'",
        "displayCanvas",
        "frame_index",
        "frame_data",
        "buffer_delay_ms",
        "console.warn",
    ]
    for token in required_tokens:
        assert token in js_content, f"Missing expected token in video_tracker.js: {token}"

    assert hasattr(VideoTracker, "__init__")
