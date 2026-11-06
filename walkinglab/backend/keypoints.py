"""
COCO 17-keypoint schema, color palette, skeleton connectivity, and export serialization utilities
(including Ultralytics YOLO-Pose dataset packaging with paired frame images and label files).
"""
import base64
import io
import json
import os
from pathlib import Path
import zipfile

import cv2
import numpy as np

DEFAULT_KEYPOINT_COLOR = "#3B82F6"

COCO_KEYPOINTS: list[dict] = [
    {"id": 0, "key": "nose", "name": "Nose", "color": "#EF4444", "group": "head"},
    {"id": 1, "key": "left_eye", "name": "L. Eye", "color": "#F97316", "group": "head"},
    {"id": 2, "key": "right_eye", "name": "R. Eye", "color": "#F59E0B", "group": "head"},
    {"id": 3, "key": "left_ear", "name": "L. Ear", "color": "#EAB308", "group": "head"},
    {"id": 4, "key": "right_ear", "name": "R. Ear", "color": "#84CC16", "group": "head"},
    {
        "id": 5,
        "key": "left_shoulder",
        "name": "L. Shoulder",
        "color": "#22C55E",
        "group": "upper_body",
    },
    {
        "id": 6,
        "key": "right_shoulder",
        "name": "R. Shoulder",
        "color": "#10B981",
        "group": "upper_body",
    },
    {
        "id": 7,
        "key": "left_elbow",
        "name": "L. Elbow",
        "color": "#14B8A6",
        "group": "upper_body",
    },
    {
        "id": 8,
        "key": "right_elbow",
        "name": "R. Elbow",
        "color": "#06B6D4",
        "group": "upper_body",
    },
    {
        "id": 9,
        "key": "left_wrist",
        "name": "L. Wrist",
        "color": "#0EA5E9",
        "group": "upper_body",
    },
    {
        "id": 10,
        "key": "right_wrist",
        "name": "R. Wrist",
        "color": "#3B82F6",
        "group": "upper_body",
    },
    {"id": 11, "key": "left_hip", "name": "L. Hip", "color": "#6366F1", "group": "torso"},
    {"id": 12, "key": "right_hip", "name": "R. Hip", "color": "#8B5CF6", "group": "torso"},
    {
        "id": 13,
        "key": "left_knee",
        "name": "L. Knee",
        "color": "#A855F7",
        "group": "lower_body",
    },
    {
        "id": 14,
        "key": "right_knee",
        "name": "R. Knee",
        "color": "#D946EF",
        "group": "lower_body",
    },
    {
        "id": 15,
        "key": "left_ankle",
        "name": "L. Ankle",
        "color": "#EC4899",
        "group": "lower_body",
    },
    {
        "id": 16,
        "key": "right_ankle",
        "name": "R. Ankle",
        "color": "#F43F5E",
        "group": "lower_body",
    },
]

KEYPOINT_BY_ID: dict[int, dict] = {kp["id"]: kp for kp in COCO_KEYPOINTS}
KEYPOINT_COLORS: dict[int, str] = {kp["id"]: kp["color"] for kp in COCO_KEYPOINTS}
KEYPOINT_NAMES: dict[int, str] = {kp["id"]: kp["name"] for kp in COCO_KEYPOINTS}

SKELETON_EDGES: list[tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def get_keypoint_color(keypoint_id: int) -> str:
    """
    Returns the unique hex color string assigned to a COCO keypoint ID.

    Parameters
    ----------
    keypoint_id : int
        The integer keypoint index (0 through 16).

    Returns
    -------
    str
        Hex color string in ``#RRGGBB`` format, falling back to ``#3B82F6`` if out of range.
    """
    return KEYPOINT_COLORS.get(keypoint_id, DEFAULT_KEYPOINT_COLOR)


def format_annotations_for_export(
    all_frames_markers: list[list[dict]], fps: float = 30.0
) -> list[dict]:
    """
    Converts per-frame marker lists into the standard CV pose estimation output schema.

    Parameters
    ----------
    all_frames_markers : list of list of dict
        Per-frame lists of marker dictionaries containing ``id``, ``x``, ``y``,
        and optional ``visibility``.
    fps : float, optional
        Video frame rate used to compute ``timestamp_ms``, by default 30.0.

    Returns
    -------
    list of dict
        List of frame dictionaries conforming to the standard pose output schema:
        ``{"frame_idx": int, "timestamp_ms": float, "keypoints": {key: {"id": int, "x": float, "y": float, "visibility": float}}}``.
    """
    effective_fps = float(fps) if fps and fps > 0 else 30.0
    exported_frames: list[dict] = []

    for frame_idx, frame_markers in enumerate(all_frames_markers or []):
        keypoints_dict: dict[str, dict] = {}
        for marker in frame_markers or []:
            if not isinstance(marker, dict) or "id" not in marker:
                continue
            kp_id = int(marker["id"])
            if kp_id not in KEYPOINT_BY_ID:
                continue
            kp_meta = KEYPOINT_BY_ID[kp_id]
            keypoints_dict[kp_meta["key"]] = {
                "id": kp_id,
                "x": round(float(marker.get("x", 0.0)), 6),
                "y": round(float(marker.get("y", 0.0)), 6),
                "visibility": round(float(marker.get("visibility", 1.0)), 4),
            }

        exported_frames.append(
            {
                "frame_idx": frame_idx,
                "timestamp_ms": round((frame_idx / effective_fps) * 1000.0, 2),
                "keypoints": keypoints_dict,
            }
        )

    return exported_frames


def format_frame_yolo_pose_label(
    frame_markers: list[dict],
    class_id: int = 0,
    bbox_padding: float = 0.08,
) -> str:
    """
    Converts a single frame's keypoint markers into an Ultralytics YOLO-Pose label string.

    The YOLO-Pose format for 17 keypoints with ``(x, y, visibility)`` is:
    ``<class_id> <x_center> <y_center> <width> <height> <px_0> <py_0> <v_0> ... <px_16> <py_16> <v_16>``
    where ``v_i`` is ``0`` (unlabeled), ``1`` (labeled but occluded), or ``2`` (labeled and visible).

    Parameters
    ----------
    frame_markers : list of dict
        List of marker dictionaries ``{"id": int, "x": float, "y": float, "visibility": float}``.
    class_id : int, optional
        Target YOLO class index (``0`` for person), by default 0.
    bbox_padding : float, optional
        Proportional margin added around the keypoint bounding box, by default 0.08.

    Returns
    -------
    str
        Single-line YOLO-Pose annotation string, or ``""`` if no valid keypoints exist.
    """
    marker_by_id: dict[int, dict] = {}
    for marker in frame_markers or []:
        if not isinstance(marker, dict) or "id" not in marker:
            continue
        kp_id = int(marker["id"])
        if kp_id in KEYPOINT_BY_ID:
            marker_by_id[kp_id] = marker

    if not marker_by_id:
        return ""

    xs: list[float] = []
    ys: list[float] = []
    kpt_tokens: list[str] = []

    for kp_id in range(len(COCO_KEYPOINTS)):
        if kp_id not in marker_by_id:
            kpt_tokens.append("0.000000 0.000000 0")
            continue

        marker = marker_by_id[kp_id]
        clamped_x = min(1.0, max(0.0, float(marker.get("x", 0.0))))
        clamped_y = min(1.0, max(0.0, float(marker.get("y", 0.0))))
        vis_raw = float(marker.get("visibility", 1.0))
        yolo_v = 1 if vis_raw <= 0.0 else 2

        xs.append(clamped_x)
        ys.append(clamped_y)
        kpt_tokens.append(f"{clamped_x:.6f} {clamped_y:.6f} {yolo_v}")

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    pad_x = max(0.04, (max_x - min_x) * float(bbox_padding))
    pad_y = max(0.04, (max_y - min_y) * float(bbox_padding))

    x0 = max(0.0, min_x - pad_x)
    x1 = min(1.0, max_x + pad_x)
    y0 = max(0.0, min_y - pad_y)
    y1 = min(1.0, max_y + pad_y)

    width = max(0.02, min(1.0, x1 - x0))
    height = max(0.02, min(1.0, y1 - y0))
    x_center = min(1.0, max(0.0, (x0 + x1) / 2.0))
    y_center = min(1.0, max(0.0, (y0 + y1) / 2.0))

    header = f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    return f"{header} {' '.join(kpt_tokens)}"


def generate_yolo_pose_data_yaml() -> str:
    """
    Generates an Ultralytics YOLO-Pose ``data.yaml`` configuration string for 17 COCO keypoints.

    Returns
    -------
    str
        YAML configuration content ready for ``yolo pose train data=data.yaml``.
    """
    return (
        "path: .\n"
        "train: images/train\n"
        "val: images/train\n"
        "kpt_shape: [17, 3]\n"
        "flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]\n"
        "names:\n"
        "  0: person\n"
    )


def _decode_b64_to_jpeg_bytes(b64_str: str) -> bytes | None:
    """
    Decodes a base64 image string (or data URL) into JPEG bytes.

    Parameters
    ----------
    b64_str : str
        Base64-encoded image string, optionally prefixed with ``data:image/...;base64,``.

    Returns
    -------
    bytes or None
        Encoded JPEG image bytes, or ``None`` if decoding fails.
    """
    if not b64_str or not isinstance(b64_str, str):
        return None
    try:
        payload = b64_str.split(",", 1)[1] if "," in b64_str else b64_str
        raw_bytes = base64.b64decode(payload)
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        ok, enc = cv2.imencode(".jpg", img)
        return enc.tobytes() if ok else None
    except Exception:
        return None


def build_yolo_pose_dataset_zip(
    all_frames_markers: list[list[dict]],
    video_path: str | None = None,
    frame_images_b64: dict[int, str] | None = None,
    video_name: str = "annotations",
    fps: float = 30.0,
) -> bytes:
    """
    Builds an in-memory ZIP archive containing a ready-to-train Ultralytics YOLO-Pose dataset.

    Each annotated frame with at least one valid keypoint is exported as a paired
    image (``images/train/<stem>.jpg``) and label file (``labels/train/<stem>.txt``),
    alongside ``data.yaml`` and ``annotations.json``.

    Parameters
    ----------
    all_frames_markers : list of list of dict
        Per-frame lists of marker dictionaries.
    video_path : str, optional
        Filesystem path to the source video file used to extract raw frames, by default None.
    frame_images_b64 : dict of (int, str), optional
        Mapping of frame indices to base64-encoded frame images captured on the client,
        by default None.
    video_name : str, optional
        Base filename stem used for exported frame images and labels, by default ``"annotations"``.
    fps : float, optional
        Frame rate used for ``annotations.json`` timestamps, by default 30.0.

    Returns
    -------
    bytes
        Raw bytes of the constructed ``.zip`` archive.
    """
    safe_stem = Path(video_name).stem if video_name else "annotations"
    if not safe_stem:
        safe_stem = "annotations"

    b64_lookup: dict[int, str] = {}
    if isinstance(frame_images_b64, dict):
        for k, v in frame_images_b64.items():
            try:
                if isinstance(v, str) and v:
                    b64_lookup[int(k)] = v
            except (ValueError, TypeError):
                continue

    cap: cv2.VideoCapture | None = None
    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)

    blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, blank_encoded = cv2.imencode(".jpg", blank_img)
    fallback_jpeg_bytes = blank_encoded.tobytes()

    buffer = io.BytesIO()
    try:
        with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("data.yaml", generate_yolo_pose_data_yaml())

            exported_json = format_annotations_for_export(all_frames_markers, fps=fps)
            zf.writestr("annotations.json", json.dumps(exported_json, indent=2))

            current_cap_pos = 0
            for frame_idx, frame_markers in enumerate(all_frames_markers or []):
                label_line = format_frame_yolo_pose_label(frame_markers)
                if not label_line:
                    continue

                item_stem = f"{safe_stem}_frame_{frame_idx:06d}"
                zf.writestr(f"labels/train/{item_stem}.txt", label_line + "\n")

                jpeg_bytes: bytes | None = None

                # 1. Check client-captured base64 image for this frame
                if frame_idx in b64_lookup:
                    jpeg_bytes = _decode_b64_to_jpeg_bytes(b64_lookup[frame_idx])

                # 2. Extract frame from source video file on disk
                if jpeg_bytes is None and cap is not None and cap.isOpened():
                    if current_cap_pos != frame_idx:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        current_cap_pos = frame_idx
                    ret, frame_bgr = cap.read()
                    if ret and frame_bgr is not None:
                        current_cap_pos += 1
                        ok, enc = cv2.imencode(".jpg", frame_bgr)
                        if ok:
                            jpeg_bytes = enc.tobytes()

                # 3. Fallback blank frame so every label has a paired image
                if jpeg_bytes is None:
                    jpeg_bytes = fallback_jpeg_bytes

                zf.writestr(f"images/train/{item_stem}.jpg", jpeg_bytes)
    finally:
        if cap is not None:
            cap.release()

    return buffer.getvalue()
