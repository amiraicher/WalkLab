"""
Automated unit and integration tests for the 17-keypoint palette, VideoTracker
annotation controls, Vue frontend component, and AnnotationController state
synchronization.
"""
import base64
import io
import json
import re
import zipfile
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

import walkinglab.backend.inference as inference
from walkinglab.backend.keypoints import (
    COCO_KEYPOINTS,
    KEYPOINT_COLORS,
    SKELETON_EDGES,
    build_yolo_pose_dataset_zip,
    format_annotations_for_export,
    format_frame_yolo_pose_label,
    generate_yolo_pose_data_yaml,
    get_keypoint_color,
)
from walkinglab.backend.training import (
    pack_session_to_zip,
    parse_frame_yolo_pose_label,
    run_few_shot_training,
    unpack_session_from_zip,
)
from walkinglab.ui.annotation import AnnotationController
from walkinglab.ui.components.video_tracker import VideoTracker

HEX_COLOR_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}$")
VIDEO_TRACKER_JS_PATH = (
    Path(__file__).resolve().parent.parent
    / "walkinglab"
    / "ui"
    / "components"
    / "video_tracker.js"
)


def _make_video_tracker() -> VideoTracker:
    """
    Creates a ``VideoTracker`` instance suitable for unit testing without
    requiring an active browser websocket connection.

    Returns
    -------
    VideoTracker
        An initialized ``VideoTracker`` element with ``run_method`` mocked
        and public control methods wrapped for call assertions.
    """
    try:
        tracker = VideoTracker()
    except Exception:
        tracker = VideoTracker.__new__(VideoTracker)
    tracker.run_method = MagicMock()

    for method_name in (
        "set_keypoint_visibility",
        "set_all_keypoints_visibility",
        "set_active_keypoint",
        "set_frame_keypoint_state",
        "step_frame",
        "toggle_playback",
        "set_batch_markers",
        "save_current_frame_to_session",
        "load_session_samples",
        "select_session_sample",
        "step_session_sample",
        "delete_session_sample",
        "exit_session_review",
        "open_file_dialog",
        "save_session_zip",
        "open_session_dialog",
        "request_session_zip_for_training",
    ):
        if hasattr(tracker, method_name):
            real_method = getattr(tracker, method_name)
            setattr(tracker, method_name, MagicMock(wraps=real_method))

    return tracker


def test_coco_keypoints_palette_and_uniqueness():
    """
    Verifies that ``COCO_KEYPOINTS`` contains 17 sequential keypoints (0..16)
    with unique keys, unique display names, and 17 unique ``#RRGGBB`` hex colors.
    """
    assert len(COCO_KEYPOINTS) == 17

    ids = [kp["id"] for kp in COCO_KEYPOINTS]
    keys = [kp["key"] for kp in COCO_KEYPOINTS]
    names = [kp["name"] for kp in COCO_KEYPOINTS]
    colors = [kp["color"] for kp in COCO_KEYPOINTS]

    assert ids == list(range(17))
    assert len(set(keys)) == 17
    assert len(set(names)) == 17
    assert len(set(colors)) == 17
    assert len(set(KEYPOINT_COLORS.values())) == 17

    for kp in COCO_KEYPOINTS:
        assert HEX_COLOR_PATTERN.match(kp["color"]), (
            f"Invalid hex color {kp['color']} for keypoint {kp['key']}"
        )
        assert re.match(r"^[a-z_]+$", kp["key"]), (
            f"Keypoint key {kp['key']} must be snake_case"
        )
        assert KEYPOINT_COLORS[kp["id"]] == kp["color"]


def test_skeleton_edges_and_get_keypoint_color():
    """
    Verifies that ``SKELETON_EDGES`` references valid keypoint IDs in ``0..16``
    and ``get_keypoint_color`` returns exact colors or a valid hex fallback.
    """
    assert len(SKELETON_EDGES) > 0
    for edge in SKELETON_EDGES:
        assert len(edge) == 2
        start_id, end_id = edge
        assert 0 <= start_id <= 16
        assert 0 <= end_id <= 16
        assert start_id != end_id

    for kp_id in range(17):
        color = get_keypoint_color(kp_id)
        assert color == KEYPOINT_COLORS[kp_id]
        assert HEX_COLOR_PATTERN.match(color)

    fallback_neg = get_keypoint_color(-1)
    fallback_oob = get_keypoint_color(999)
    assert HEX_COLOR_PATTERN.match(fallback_neg)
    assert HEX_COLOR_PATTERN.match(fallback_oob)


def test_format_annotations_for_export_schema():
    """
    Verifies that ``format_annotations_for_export`` produces the exact dictionary
    schema mandated by ``cv-pose-guidelines`` (``frame_idx``, ``timestamp_ms``,
    and ``keypoints`` keyed by ``snake_case`` joint name with ``x``, ``y``, and
    ``visibility``).
    """
    sample_frames = [
        [
            {"id": 0, "x": 0.5, "y": 0.2, "visibility": 1.0},
            {"id": 15, "x": 0.45, "y": 0.88, "visibility": 0.0},
        ],
        [
            {"id": 16, "x": 0.55, "y": 0.90},
        ],
    ]

    exported = format_annotations_for_export(sample_frames, fps=25.0)
    assert len(exported) == 2

    frame_0 = exported[0]
    assert frame_0["frame_idx"] == 0
    assert pytest.approx(frame_0["timestamp_ms"]) == 0.0
    assert "keypoints" in frame_0
    assert "nose" in frame_0["keypoints"]
    assert "left_ankle" in frame_0["keypoints"]
    assert frame_0["keypoints"]["nose"]["x"] == pytest.approx(0.5)
    assert frame_0["keypoints"]["nose"]["y"] == pytest.approx(0.2)
    assert frame_0["keypoints"]["nose"]["visibility"] == pytest.approx(1.0)
    assert frame_0["keypoints"]["left_ankle"]["x"] == pytest.approx(0.45)
    assert frame_0["keypoints"]["left_ankle"]["y"] == pytest.approx(0.88)
    assert frame_0["keypoints"]["left_ankle"]["visibility"] == pytest.approx(0.0)

    frame_1 = exported[1]
    assert frame_1["frame_idx"] == 1
    assert pytest.approx(frame_1["timestamp_ms"]) == 40.0
    assert "right_ankle" in frame_1["keypoints"]
    assert frame_1["keypoints"]["right_ankle"]["x"] == pytest.approx(0.55)
    assert frame_1["keypoints"]["right_ankle"]["y"] == pytest.approx(0.90)
    assert frame_1["keypoints"]["right_ankle"]["visibility"] == pytest.approx(1.0)


def test_video_tracker_annotation_methods():
    """
    Verifies that ``VideoTracker`` exposes all required annotation control
    methods and dispatches the expected payloads through ``run_method``.
    """
    required_methods = (
        "set_keypoint_visibility",
        "set_all_keypoints_visibility",
        "set_active_keypoint",
        "set_frame_keypoint_state",
        "step_frame",
        "toggle_playback",
    )
    for method_name in required_methods:
        assert hasattr(VideoTracker, method_name), (
            f"VideoTracker missing method {method_name}"
        )

    tracker = _make_video_tracker()

    tracker.set_keypoint_visibility(5, False)
    tracker.run_method.assert_called_with(
        "set_keypoint_visibility", {"id": 5, "visible": False}
    )

    tracker.set_all_keypoints_visibility(True)
    tracker.run_method.assert_called_with(
        "set_all_keypoints_visibility", {"visible": True}
    )

    tracker.set_active_keypoint(15)
    tracker.run_method.assert_called_with("set_active_keypoint", {"id": 15})

    tracker.set_active_keypoint(None)
    tracker.run_method.assert_called_with("set_active_keypoint", {"id": None})

    tracker.set_frame_keypoint_state(15, "occluded")
    tracker.run_method.assert_called_with(
        "set_frame_keypoint_state", {"id": 15, "state": "occluded"}
    )

    with pytest.raises(ValueError):
        tracker.set_frame_keypoint_state(15, "invalid_state")

    tracker.step_frame(-1)
    tracker.run_method.assert_called_with("step_frame", {"delta": -1})

    tracker.toggle_playback()
    tracker.run_method.assert_called_with("toggle_playback")

    tracker.save_session_zip()
    tracker.run_method.assert_called_with("save_session_zip")

    tracker.open_session_dialog()
    tracker.run_method.assert_called_with("open_session_dialog")

    tracker.request_session_zip_for_training()
    tracker.run_method.assert_called_with("request_session_zip_for_training")


def test_video_tracker_js_palette_and_interactivity():
    """
    Verifies that ``video_tracker.js`` defines the 17-keypoint color map,
    skeleton topology, visibility state, interactive drag-and-drop handlers,
    and ``annotation_updated`` event emission.
    """
    assert VIDEO_TRACKER_JS_PATH.exists()
    js_content = VIDEO_TRACKER_JS_PATH.read_text(encoding="utf-8")

    for kp in COCO_KEYPOINTS:
        assert kp["color"].lower() in js_content.lower(), (
            f"Hex color {kp['color']} for keypoint {kp['key']} missing from video_tracker.js"
        )

    required_js_identifiers = (
        "keypointColors",
        "skeletonEdges",
        "visibilityMap",
        "activeKeypointId",
        "set_keypoint_visibility",
        "set_all_keypoints_visibility",
        "set_active_keypoint",
        "set_frame_keypoint_state",
        "step_frame",
        "toggle_playback",
        "startDrag",
        "onMouseMove",
        "stopDrag",
        "handleCanvasMouseDown",
        "annotation_updated",
        "captureCurrentFrameBase64",
    )
    for identifier in required_js_identifiers:
        assert identifier in js_content, (
            f"Identifier '{identifier}' missing from video_tracker.js"
        )


def test_annotation_controller_visibility_and_selection():
    """
    Verifies ``AnnotationController`` state initialization, per-keypoint and
    bulk visibility toggling, and active keypoint selection/deselection.
    """
    mock_tracker = _make_video_tracker()
    with patch("walkinglab.ui.annotation.ui.notify"):
        controller = AnnotationController(tracker=mock_tracker)

        assert len(controller.visibility_state) == 17
        assert all(controller.visibility_state[kp_id] is True for kp_id in range(17))
        assert controller.active_keypoint_id is None

        new_state = controller.toggle_keypoint_visibility(11)
        assert new_state is False
        assert controller.visibility_state[11] is False
        mock_tracker.set_keypoint_visibility.assert_called_with(11, False)

        restored_state = controller.toggle_keypoint_visibility(11)
        assert restored_state is True
        assert controller.visibility_state[11] is True
        mock_tracker.set_keypoint_visibility.assert_called_with(11, True)

        controller.set_all_visibility(False)
        assert all(controller.visibility_state[kp_id] is False for kp_id in range(17))
        mock_tracker.set_all_keypoints_visibility.assert_called_with(False)

        controller.set_all_visibility(True)
        assert all(controller.visibility_state[kp_id] is True for kp_id in range(17))
        mock_tracker.set_all_keypoints_visibility.assert_called_with(True)

        selected = controller.select_keypoint_for_edit(15)
        assert selected == 15
        assert controller.active_keypoint_id == 15
        mock_tracker.set_active_keypoint.assert_called_with(15)

        deselected = controller.select_keypoint_for_edit(15)
        assert deselected is None
        assert controller.active_keypoint_id is None
        mock_tracker.set_active_keypoint.assert_called_with(None)


def test_annotation_controller_frame_edits_and_export():
    """
    Verifies that ``AnnotationController`` synchronizes frame-level edits,
    occlusion states, deletions, and JSON export formatting according to
    ``cv-pose-guidelines``.
    """
    mock_tracker = _make_video_tracker()
    with patch("walkinglab.ui.annotation.ui.notify"):
        controller = AnnotationController(tracker=mock_tracker)

        initial_frames = [
            [
                {"id": 0, "x": 0.50, "y": 0.15, "visibility": 1.0},
                {"id": 15, "x": 0.42, "y": 0.85, "visibility": 1.0},
                {"id": 16, "x": 0.58, "y": 0.86, "visibility": 1.0},
            ],
            [
                {"id": 15, "x": 0.44, "y": 0.84, "visibility": 1.0},
            ],
        ]
        controller.all_frames_markers = initial_frames
        controller.video_name = "walk_test"
        assert controller.current_frame_index == 0
        assert len(controller.all_frames_markers) == 2

        # Mark left_ankle (15) as occluded on frame 0
        controller.set_keypoint_frame_state(15, "occluded")
        frame_0_by_id = {m["id"]: m for m in controller.all_frames_markers[0]}
        assert frame_0_by_id[15]["visibility"] == 0.0
        mock_tracker.set_frame_keypoint_state.assert_called_with(15, "occluded")

        # Mark left_ankle (15) back to visible on frame 0
        controller.set_keypoint_frame_state(15, "visible")
        frame_0_by_id = {m["id"]: m for m in controller.all_frames_markers[0]}
        assert frame_0_by_id[15]["visibility"] == 1.0

        # Delete right_ankle (16) from frame 0
        controller.set_keypoint_frame_state(16, "delete")
        frame_0_ids = {m["id"] for m in controller.all_frames_markers[0]}
        assert 16 not in frame_0_ids

        # Simulate client-side drag update on frame 1
        controller.handle_annotation_updated(
            {
                "frame_index": 1,
                "markers": [
                    {"id": 15, "x": 0.48, "y": 0.82, "visibility": 0.0},
                ],
            }
        )
        assert controller.current_frame_index == 1
        assert controller.all_frames_markers[1] == [
            {"id": 15, "x": 0.48, "y": 0.82, "visibility": 0.0}
        ]

        # Verify JSON export matches cv-pose-guidelines schema
        json_str = controller.export_json(fps=30.0)
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)
        assert len(parsed) == 2

        assert parsed[0]["frame_idx"] == 0
        assert "nose" in parsed[0]["keypoints"]
        assert "left_ankle" in parsed[0]["keypoints"]
        assert "right_ankle" not in parsed[0]["keypoints"]

        assert parsed[1]["frame_idx"] == 1
        assert pytest.approx(parsed[1]["timestamp_ms"], rel=1e-2) == 1000.0 / 30.0
        assert parsed[1]["keypoints"]["left_ankle"]["x"] == pytest.approx(0.48)
        assert parsed[1]["keypoints"]["left_ankle"]["y"] == pytest.approx(0.82)
        assert parsed[1]["keypoints"]["left_ankle"]["visibility"] == pytest.approx(0.0)


def test_format_frame_yolo_pose_label():
    """
    Verifies that ``format_frame_yolo_pose_label`` formats a frame's markers
    into the 56-token Ultralytics YOLO-Pose annotation line with correct
    bounding box and keypoint visibility flags (v=2 visible, v=1 occluded,
    v=0 unlabeled).
    """
    assert format_frame_yolo_pose_label([]) == ""

    markers = [
        {"id": 0, "x": 0.50, "y": 0.20, "visibility": 1.0},   # Nose: visible (v=2)
        {"id": 15, "x": 0.40, "y": 0.80, "visibility": 0.0},  # L. Ankle: occluded (v=1)
        {"id": 16, "x": 0.60, "y": 0.85, "visibility": 1.0},  # R. Ankle: visible (v=2)
    ]

    line = format_frame_yolo_pose_label(markers)
    tokens = line.strip().split()
    assert len(tokens) == 5 + 17 * 3  # 56 tokens

    class_id = int(tokens[0])
    x_center, y_center, width, height = (float(t) for t in tokens[1:5])

    assert class_id == 0
    assert 0.0 <= x_center <= 1.0
    assert 0.0 <= y_center <= 1.0
    assert 0.0 < width <= 1.0
    assert 0.0 < height <= 1.0

    # Bounding box must enclose the annotated keypoints (x in [0.40, 0.60], y in [0.20, 0.85])
    assert (x_center - width / 2.0) <= 0.40 + 1e-5
    assert (x_center + width / 2.0) >= 0.60 - 1e-5
    assert (y_center - height / 2.0) <= 0.20 + 1e-5
    assert (y_center + height / 2.0) >= 0.85 - 1e-5

    # Check keypoint 0 (Nose): visible -> v=2
    kp0_x, kp0_y, kp0_v = float(tokens[5]), float(tokens[6]), int(tokens[7])
    assert pytest.approx(kp0_x) == 0.50
    assert pytest.approx(kp0_y) == 0.20
    assert kp0_v == 2

    # Check keypoint 1 (L. Eye): missing -> 0 0 0
    kp1_x, kp1_y, kp1_v = float(tokens[8]), float(tokens[9]), int(tokens[10])
    assert kp1_x == 0.0
    assert kp1_y == 0.0
    assert kp1_v == 0

    # Check keypoint 15 (L. Ankle): occluded -> v=1
    kp15_base = 5 + 15 * 3
    kp15_x, kp15_y, kp15_v = (
        float(tokens[kp15_base]),
        float(tokens[kp15_base + 1]),
        int(tokens[kp15_base + 2]),
    )
    assert pytest.approx(kp15_x) == 0.40
    assert pytest.approx(kp15_y) == 0.80
    assert kp15_v == 1

    # Check keypoint 16 (R. Ankle): visible -> v=2
    kp16_base = 5 + 16 * 3
    kp16_x, kp16_y, kp16_v = (
        float(tokens[kp16_base]),
        float(tokens[kp16_base + 1]),
        int(tokens[kp16_base + 2]),
    )
    assert pytest.approx(kp16_x) == 0.60
    assert pytest.approx(kp16_y) == 0.85
    assert kp16_v == 2


def test_generate_yolo_pose_data_yaml():
    """
    Verifies that ``generate_yolo_pose_data_yaml`` outputs a valid Ultralytics
    YOLO-Pose dataset configuration containing ``kpt_shape: [17, 3]``,
    ``train: images/train``, ``flip_idx``, and ``0: person``.
    """
    yaml_str = generate_yolo_pose_data_yaml()
    assert "kpt_shape: [17, 3]" in yaml_str
    assert "train: images/train" in yaml_str
    assert "flip_idx:" in yaml_str
    assert "0: person" in yaml_str


def test_build_and_export_yolo_pose_dataset_zip(tmp_path: Path):
    """
    Verifies that ``build_yolo_pose_dataset_zip`` and
    ``AnnotationController.export_yolo_dataset_zip`` package extracted frame
    JPEG images, ``labels/train/*.txt`` files, ``data.yaml``, and
    ``annotations.json`` into a valid ZIP archive across video file, base64
    canvas snapshot, and fallback image modes.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory fixture.
    """
    video_path = tmp_path / "sample_gait.mp4"
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (width, height))
    assert writer.isOpened()
    for frame_idx in range(3):
        img = np.full((height, width, 3), fill_value=40 + frame_idx * 50, dtype=np.uint8)
        writer.write(img)
    writer.release()

    all_frames_markers = [
        [
            {"id": 0, "x": 0.50, "y": 0.20, "visibility": 1.0},
            {"id": 15, "x": 0.45, "y": 0.85, "visibility": 1.0},
        ],
        [],  # Empty unannotated frame should be skipped
        [
            {"id": 16, "x": 0.55, "y": 0.88, "visibility": 0.0},
        ],
    ]

    # 1. Test with real OpenCV video file
    zip_bytes_video = build_yolo_pose_dataset_zip(
        all_frames_markers=all_frames_markers,
        video_name="sample_gait",
        video_path=video_path,
        fps=30.0,
    )
    with zipfile.ZipFile(io.BytesIO(zip_bytes_video), "r") as zf:
        names = set(zf.namelist())
        assert "data.yaml" in names
        assert "annotations.json" in names
        assert "images/train/sample_gait_frame_000000.jpg" in names
        assert "labels/train/sample_gait_frame_000000.txt" in names
        assert "images/train/sample_gait_frame_000002.jpg" in names
        assert "labels/train/sample_gait_frame_000002.txt" in names
        assert "images/train/sample_gait_frame_000001.jpg" not in names

        jpg_bytes = zf.read("images/train/sample_gait_frame_000000.jpg")
        assert jpg_bytes.startswith(b"\xff\xd8")

        label_0 = zf.read("labels/train/sample_gait_frame_000000.txt").decode("utf-8").strip()
        assert len(label_0.split()) == 56

    # 2. Test via AnnotationController with base64 canvas snapshots and fallback
    dummy_canvas = np.full((120, 160, 3), fill_value=180, dtype=np.uint8)
    ok, enc = cv2.imencode(".jpg", dummy_canvas)
    assert ok
    b64_data_url = "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode("ascii")

    mock_tracker = _make_video_tracker()
    with patch("walkinglab.ui.annotation.ui.notify"):
        controller = AnnotationController(tracker=mock_tracker)
        controller.video_name = "webcam_session"
        controller.handle_annotation_updated(
            {
                "frame_index": 0,
                "markers": [{"id": 0, "x": 0.5, "y": 0.25, "visibility": 1.0}],
                "frame_image_b64": b64_data_url,
            }
        )
        controller.handle_annotation_updated(
            {
                "frame_index": 1,
                "markers": [{"id": 15, "x": 0.48, "y": 0.82, "visibility": 0.0}],
            }
        )

        zip_bytes_ctrl = controller.export_yolo_dataset_zip(fps=30.0)

    with zipfile.ZipFile(io.BytesIO(zip_bytes_ctrl), "r") as zf:
        names = set(zf.namelist())
        assert "data.yaml" in names
        assert "annotations.json" in names
        assert "images/train/webcam_session_frame_000000.jpg" in names
        assert "labels/train/webcam_session_frame_000000.txt" in names
        assert "images/train/webcam_session_frame_000001.jpg" in names
        assert "labels/train/webcam_session_frame_000001.txt" in names

        for img_name in (
            "images/train/webcam_session_frame_000000.jpg",
            "images/train/webcam_session_frame_000001.jpg",
        ):
            img_data = zf.read(img_name)
            assert img_data.startswith(b"\xff\xd8")
            decoded = cv2.imdecode(np.frombuffer(img_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            assert decoded is not None and decoded.size > 0


def test_cpu_bound_pickling_safety():
    """
    Verifies that top-level functions and pure-data payloads passed to
    ``nicegui.run.cpu_bound`` can be pickled by ``ForkingPickler.dumps`` even
    when ``AnnotationController`` is bound to a live ``VideoTracker`` UI element.
    """
    mock_tracker = _make_video_tracker()
    with patch("walkinglab.ui.annotation.ui.notify"):
        controller = AnnotationController(tracker=mock_tracker)
        controller.all_frames_markers = [
            [{"id": 0, "x": 0.5, "y": 0.2, "visibility": 1.0}]
        ]
        controller.session_samples = [
            {
                "sample_id": "walk_frame_000000",
                "video_name": "walk",
                "frame_index": 0,
                "image_b64": "",
                "markers": [{"id": 0, "x": 0.5, "y": 0.2, "visibility": 1.0}],
            }
        ]

    # Top-level functions and pure Python data must pickle cleanly without pulling in NiceGUI elements
    assert ForkingPickler.dumps(pack_session_to_zip)
    assert ForkingPickler.dumps(unpack_session_from_zip)
    assert ForkingPickler.dumps(run_few_shot_training)
    assert ForkingPickler.dumps(build_yolo_pose_dataset_zip)
    assert ForkingPickler.dumps(controller.session_samples)
    assert ForkingPickler.dumps(controller.all_frames_markers)


def test_session_zip_pack_and_unpack_roundtrip():
    """
    Verifies that ``pack_session_to_zip`` and ``unpack_session_from_zip``
    preserve multi-video session samples, images, YOLO-Pose labels, and
    ``session.json`` across a full pack/unpack roundtrip, and also support
    fallback parsing of YOLO-Pose label files via ``parse_frame_yolo_pose_label``.
    """
    img1 = np.full((100, 120, 3), fill_value=90, dtype=np.uint8)
    ok1, enc1 = cv2.imencode(".jpg", img1)
    assert ok1
    b64_1 = "data:image/jpeg;base64," + base64.b64encode(enc1.tobytes()).decode("ascii")

    img2 = np.full((100, 120, 3), fill_value=200, dtype=np.uint8)
    ok2, enc2 = cv2.imencode(".jpg", img2)
    assert ok2
    b64_2 = "data:image/jpeg;base64," + base64.b64encode(enc2.tobytes()).decode("ascii")

    samples = [
        {
            "sample_id": "walk1_frame_000005",
            "video_name": "walk1",
            "frame_index": 5,
            "image_b64": b64_1,
            "markers": [
                {"id": 0, "x": 0.50, "y": 0.20, "visibility": 1.0},
                {"id": 15, "x": 0.42, "y": 0.85, "visibility": 0.0},
            ],
        },
        {
            "sample_id": "walk2_frame_000012",
            "video_name": "walk2",
            "frame_index": 12,
            "image_b64": b64_2,
            "markers": [
                {"id": 16, "x": 0.58, "y": 0.88, "visibility": 1.0},
            ],
        },
    ]

    zip_bytes = pack_session_to_zip(samples, session_name="clinical_session")
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        names = set(zf.namelist())
        assert "session.json" in names
        assert "data.yaml" in names
        assert "annotations.json" in names
        assert "images/train/walk1_frame_000005.jpg" in names
        assert "labels/train/walk1_frame_000005.txt" in names
        assert "images/train/walk2_frame_000012.jpg" in names
        assert "labels/train/walk2_frame_000012.txt" in names

        label_text = zf.read("labels/train/walk1_frame_000005.txt").decode("utf-8")
        parsed_markers = parse_frame_yolo_pose_label(label_text)
        by_id = {m["id"]: m for m in parsed_markers}
        assert 0 in by_id and by_id[0]["visibility"] == 1.0
        assert 15 in by_id and by_id[15]["visibility"] == 0.0

    unpacked = unpack_session_from_zip(zip_bytes)
    assert len(unpacked) == 2
    assert unpacked[0]["sample_id"] == "walk1_frame_000005"
    assert unpacked[0]["video_name"] == "walk1"
    assert unpacked[0]["frame_index"] == 5
    assert unpacked[0]["image_b64"].startswith("data:image/jpeg;base64,")
    assert len(unpacked[0]["markers"]) == 2

    assert unpacked[1]["sample_id"] == "walk2_frame_000012"
    assert unpacked[1]["video_name"] == "walk2"
    assert unpacked[1]["frame_index"] == 12
    assert unpacked[1]["image_b64"].startswith("data:image/jpeg;base64,")
    assert len(unpacked[1]["markers"]) == 1

    # Also verify fallback unpacking from a standard YOLO-Pose zip without session.json
    fallback_zip = build_yolo_pose_dataset_zip(
        all_frames_markers=[[{"id": 0, "x": 0.5, "y": 0.25, "visibility": 1.0}]],
        video_name="legacy_clip",
        frame_images_b64={0: b64_1},
    )
    fallback_samples = unpack_session_from_zip(fallback_zip)
    assert len(fallback_samples) == 1
    assert fallback_samples[0]["video_name"] == "legacy_clip"
    assert fallback_samples[0]["frame_index"] == 0
    assert fallback_samples[0]["image_b64"].startswith("data:image/jpeg;base64,")
    assert len(fallback_samples[0]["markers"]) == 1


def test_inference_and_learning_separation_and_few_shot_training():
    """
    Verifies strict decoupling between ``walkinglab.backend.inference`` and
    ``walkinglab.backend.training``: setting custom weights via the
    ``@property`` ``inference.config.active_weights_path`` does not instantiate
    the OpenVINO ``PoseEstimator`` when ``estimator is None``, and
    ``run_few_shot_training`` lazily invokes Ultralytics ``YOLO.train`` on the
    unpacked session dataset.
    """
    assert isinstance(
        getattr(type(inference.config), "active_weights_path", None), property
    )
    assert isinstance(
        getattr(AnnotationController, "active_weights_path", None), property
    )

    prev_estimator = inference.estimator
    prev_weights = inference.config.active_weights_path
    try:
        inference.estimator = None
        inference.config.active_weights_path = "custom_weights.pt"
        assert inference.config.active_weights_path == "custom_weights.pt"
        assert inference.get_active_weights_path() == "custom_weights.pt"
        # Must NOT eagerly instantiate PoseEstimator when setting weights path
        assert inference.estimator is None

        inference.config.active_weights_path = None
        assert inference.config.active_weights_path == "yolo26s-pose.pt"
    finally:
        inference.estimator = prev_estimator
        inference.config.active_weights_path = prev_weights

    # Empty session zip must raise ValueError
    empty_zip = pack_session_to_zip([], session_name="empty")
    with pytest.raises(ValueError):
        run_few_shot_training(empty_zip)

    # Non-empty session zip with mocked Ultralytics YOLO trainer
    img = np.full((80, 80, 3), fill_value=128, dtype=np.uint8)
    ok, enc = cv2.imencode(".jpg", img)
    assert ok
    b64_img = "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode("ascii")

    valid_zip = pack_session_to_zip(
        [
            {
                "sample_id": "subj1_frame_000001",
                "video_name": "subj1",
                "frame_index": 1,
                "image_b64": b64_img,
                "markers": [{"id": 0, "x": 0.5, "y": 0.2, "visibility": 1.0}],
            }
        ],
        session_name="train_session",
    )

    mock_yolo_cls = MagicMock()
    mock_model_instance = MagicMock()

    def fake_train(**kwargs):
        project_dir = Path(kwargs["project"])
        weights_dir = project_dir / kwargs["name"] / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        (weights_dir / "best.pt").write_bytes(b"FAKE_FINE_TUNED_PT_WEIGHTS")
        data_yaml_path = Path(kwargs["data"])
        assert data_yaml_path.exists()
        assert (data_yaml_path.parent / "images" / "train" / "subj1_frame_000001.jpg").exists()
        assert (data_yaml_path.parent / "labels" / "train" / "subj1_frame_000001.txt").exists()

    mock_model_instance.train.side_effect = fake_train
    mock_yolo_cls.return_value = mock_model_instance

    with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_cls)}):
        result = run_few_shot_training(
            session_zip_bytes=valid_zip,
            base_weights_path="custom_base.pt",
            epochs=15,
            lr0=0.0005,
            freeze_layers=8,
            imgsz=640,
        )

    mock_yolo_cls.assert_called_once_with("custom_base.pt")
    assert mock_model_instance.train.called
    train_kwargs = mock_model_instance.train.call_args.kwargs
    assert train_kwargs["epochs"] == 15
    assert train_kwargs["lr0"] == pytest.approx(0.0005)
    assert train_kwargs["freeze"] == 8
    assert result["weights_bytes"] == b"FAKE_FINE_TUNED_PT_WEIGHTS"
    assert result["epochs"] == 15
    assert result["num_samples"] == 1


def test_annotation_controller_and_video_tracker_session_workflow():
    """
    Verifies the end-to-end session workflow across ``VideoTracker``,
    ``video_tracker.js``, and ``AnnotationController`` (saving annotated frames
    to session, selecting/stepping/deleting filmstrip samples, packing/loading
    compressed session archives, and updating model weights).
    """
    annotation_py_path = (
        Path(__file__).resolve().parent.parent / "walkinglab" / "ui" / "annotation.py"
    )
    annotation_py_content = annotation_py_path.read_text(encoding="utf-8")
    assert "Export YOLO Dataset" not in annotation_py_content
    assert "Save Current Frame to Session (GT)" in annotation_py_content

    js_content = VIDEO_TRACKER_JS_PATH.read_text(encoding="utf-8")
    assert 'v-if="interactive"' in js_content
    assert '@click="save_current_frame_to_session"' in js_content
    for js_symbol in (
        "sessionSamples",
        "activeSessionIndex",
        "save_current_frame_to_session",
        "load_session_samples",
        "select_session_sample",
        "step_session_sample",
        "delete_session_sample",
        "exit_session_review",
        "open_file_dialog",
        "session_updated",
        "buildSessionZipUint8Array",
        "save_session_zip",
        "open_session_dialog",
        "handleSessionZipUpload",
        "request_session_zip_for_training",
        "training_zip_ready",
    ):
        assert js_symbol in js_content, f"Missing '{js_symbol}' in video_tracker.js"

    main_py_path = Path(__file__).resolve().parent.parent / "main.py"
    main_py_content = main_py_path.read_text(encoding="utf-8")
    assert "delete_on_close=False" in main_py_content
    assert "temp_video.close()" in main_py_content
    assert "set_last_uploaded_video_path" not in main_py_content

    mock_tracker = _make_video_tracker()
    img = np.full((60, 80, 3), fill_value=150, dtype=np.uint8)
    ok, enc = cv2.imencode(".jpg", img)
    assert ok
    b64_img = "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode("ascii")

    prev_estimator = inference.estimator
    prev_weights = inference.config.active_weights_path
    try:
        inference.estimator = None
        with patch("walkinglab.ui.annotation.ui.notify"):
            controller = AnnotationController(tracker=mock_tracker)
            controller.video_name = "gait_clip"
            controller.all_frames_markers = [
                [{"id": 0, "x": 0.5, "y": 0.2, "visibility": 1.0}],
                [{"id": 15, "x": 0.45, "y": 0.85, "visibility": 0.0}],
            ]
            controller.frame_images[0] = b64_img
            controller.frame_images[1] = b64_img

            # 1. Model weights selection via @property and helper
            controller.active_weights_path = "patient_v1.pt"
            assert controller.active_weights_path == "patient_v1.pt"
            assert inference.config.active_weights_path == "patient_v1.pt"

            chosen = controller.set_model_weights("patient_v2.pt")
            assert chosen == "patient_v2.pt"
            assert controller.active_weights_path == "patient_v2.pt"

            # 2. Save frame 0 and frame 1 to session
            controller.current_frame_index = 0
            s0 = controller.save_current_frame_to_session()
            assert s0["sample_id"] == "gait_clip_frame_000000"
            mock_tracker.save_current_frame_to_session.assert_called_with()

            controller.current_frame_index = 1
            s1 = controller.save_current_frame_to_session()
            assert s1["sample_id"] == "gait_clip_frame_000001"
            assert len(controller.session_samples) == 2

            # 3. Select and step through session samples
            selected = controller.select_session_sample(0)
            assert selected is not None
            assert controller.active_session_index == 0
            assert controller.in_session_review is True
            mock_tracker.select_session_sample.assert_called_with(0)

            stepped = controller.step_session_sample(1)
            assert stepped is not None
            assert controller.active_session_index == 1
            mock_tracker.step_session_sample.assert_called_with(1)

            # 4. Pack session to zip, delete a sample, and reload from zip
            session_zip = controller.pack_session_zip(session_name="rehab_session")
            controller.delete_session_sample(0)
            assert len(controller.session_samples) == 1
            mock_tracker.delete_session_sample.assert_called_with(0)

            restored = controller.load_session_zip(session_zip)
            assert len(restored) == 2
            assert len(controller.session_samples) == 2
            mock_tracker.load_session_samples.assert_called_once()

            # 5. Synchronize session_updated event from client
            controller.handle_session_updated(
                {
                    "session_samples": restored[:1],
                    "active_session_index": 0,
                    "active_mode": "session",
                }
            )
            assert len(controller.session_samples) == 1
            assert controller.active_session_index == 0
            assert controller.in_session_review is True
    finally:
        inference.estimator = prev_estimator
        inference.config.active_weights_path = prev_weights


