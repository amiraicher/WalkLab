"""
Decoupled session archive compression/decompression and few-shot YOLO-Pose fine-tuning pipeline.

This module is strictly isolated from the inference runtime: it does not import
``walkinglab.backend.inference`` and never instantiates any deep learning model at import time.
All functions are top-level and stateless so they can be dispatched safely via ``run.cpu_bound``
across ``multiprocessing`` worker processes.
"""
import base64
import io
import json
import os
from pathlib import Path
import re
import tempfile
import zipfile

import cv2
import numpy as np

from walkinglab.backend.keypoints import (
    COCO_KEYPOINTS,
    format_annotations_for_export,
    format_frame_yolo_pose_label,
    generate_yolo_pose_data_yaml,
)

DEFAULT_WEIGHTS_PATH = "yolo26s-pose.pt"

DEFAULT_FEW_SHOT_CONFIG: dict = {
    "epochs": 10,
    "lr0": 0.001,
    "freeze_layers": 10,
    "imgsz": 640,
    "batch": 4,
}


def _decode_image_b64_to_jpeg(image_b64: str | None) -> bytes:
    """
    Decodes a base64 image string (or data URI) into JPEG bytes, falling back to a
    blank 640x480 JPEG image if ``image_b64`` is missing or invalid.

    Parameters
    ----------
    image_b64 : str or None
        Base64-encoded image string, optionally prefixed with ``data:image/...;base64,``.

    Returns
    -------
    bytes
        Raw JPEG image bytes.
    """
    if image_b64 and isinstance(image_b64, str):
        try:
            payload = (
                image_b64.split(",", 1)[1] if "," in image_b64 else image_b64
            )
            raw_bytes = base64.b64decode(payload)
            arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                ok, enc = cv2.imencode(".jpg", img)
                if ok:
                    return enc.tobytes()
        except Exception:
            pass

    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    _, enc = cv2.imencode(".jpg", blank)
    return enc.tobytes()


def parse_frame_yolo_pose_label(label_text: str) -> list[dict]:
    """
    Parses a single-line Ultralytics YOLO-Pose label string back into marker dictionaries.

    Parameters
    ----------
    label_text : str
        YOLO-Pose label text containing ``<class_id> <xc> <yc> <w> <h>`` followed by
        17 ``<px_i> <py_i> <v_i>`` triples.

    Returns
    -------
    list of dict
        List of marker dictionaries ``{"id": int, "x": float, "y": float, "visibility": float}``
        for all labeled keypoints (``v_i > 0``).
    """
    if not label_text or not isinstance(label_text, str):
        return []

    lines = [line.strip() for line in label_text.splitlines() if line.strip()]
    if not lines:
        return []

    tokens = lines[0].split()
    expected_tokens = 5 + len(COCO_KEYPOINTS) * 3
    if len(tokens) < expected_tokens:
        return []

    markers: list[dict] = []
    for kp_id in range(len(COCO_KEYPOINTS)):
        base_idx = 5 + kp_id * 3
        try:
            px = float(tokens[base_idx])
            py = float(tokens[base_idx + 1])
            vis_flag = int(round(float(tokens[base_idx + 2])))
        except (ValueError, IndexError):
            continue

        if vis_flag <= 0:
            continue

        markers.append(
            {
                "id": kp_id,
                "x": round(min(1.0, max(0.0, px)), 6),
                "y": round(min(1.0, max(0.0, py)), 6),
                "visibility": 1.0 if vis_flag >= 2 else 0.0,
            }
        )

    return markers


def pack_session_to_zip(
    session_samples: list[dict],
    session_name: str = "walklab_session",
    fps: float = 30.0,
) -> bytes:
    """
    Compresses a list of annotated session samples into an Ultralytics YOLO-Pose ZIP archive.

    The resulting archive contains:
    - ``data.yaml``: Ultralytics YOLO-Pose dataset configuration.
    - ``session.json``: Session manifest preserving sample metadata and marker lists.
    - ``annotations.json``: Standardized CV pose schema export.
    - ``images/train/<sample_id>.jpg``: Clean extracted frame JPEG for each sample.
    - ``labels/train/<sample_id>.txt``: 56-token YOLO-Pose label file for each sample.

    Parameters
    ----------
    session_samples : list of dict
        List of sample dictionaries with keys ``sample_id``, ``video_name``,
        ``frame_index``, ``image_b64``, and ``markers``.
    session_name : str, optional
        Base session name recorded in ``session.json``, by default ``"walklab_session"``.
    fps : float, optional
        Frame rate used for ``annotations.json`` timestamps, by default 30.0.

    Returns
    -------
    bytes
        Compressed ``.zip`` archive bytes.
    """
    clean_session_name = Path(str(session_name or "walklab_session")).stem or "walklab_session"
    manifest_samples: list[dict] = []
    all_markers_list: list[list[dict]] = []

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("data.yaml", generate_yolo_pose_data_yaml())

        for idx, sample in enumerate(session_samples or []):
            if not isinstance(sample, dict):
                continue

            raw_video_name = str(sample.get("video_name") or clean_session_name)
            video_stem = Path(raw_video_name).stem or clean_session_name
            frame_index = int(sample.get("frame_index", idx))
            sample_id = str(
                sample.get("sample_id")
                or f"{video_stem}_frame_{frame_index:06d}"
            )
            markers = [
                dict(m)
                for m in (sample.get("markers") or [])
                if isinstance(m, dict) and "id" in m
            ]
            all_markers_list.append(markers)

            jpeg_bytes = _decode_image_b64_to_jpeg(sample.get("image_b64"))
            image_rel_path = f"images/train/{sample_id}.jpg"
            label_rel_path = f"labels/train/{sample_id}.txt"

            zf.writestr(image_rel_path, jpeg_bytes)

            label_line = format_frame_yolo_pose_label(markers)
            zf.writestr(
                label_rel_path,
                (label_line + "\n") if label_line else "",
            )

            manifest_samples.append(
                {
                    "sample_id": sample_id,
                    "video_name": video_stem,
                    "frame_index": frame_index,
                    "image_path": image_rel_path,
                    "label_path": label_rel_path,
                    "markers": markers,
                }
            )

        session_manifest = {
            "session_name": clean_session_name,
            "version": 1,
            "samples": manifest_samples,
        }
        zf.writestr("session.json", json.dumps(session_manifest, indent=2))

        exported_annotations = format_annotations_for_export(
            all_markers_list, fps=fps
        )
        zf.writestr(
            "annotations.json", json.dumps(exported_annotations, indent=2)
        )

    return buffer.getvalue()


def unpack_session_from_zip(zip_bytes: bytes) -> list[dict]:
    """
    Decompresses a saved session or YOLO-Pose dataset ZIP archive into session sample dictionaries.

    Parameters
    ----------
    zip_bytes : bytes
        Raw bytes of a ``.zip`` archive created by ``pack_session_to_zip`` or
        ``build_yolo_pose_dataset_zip``.

    Returns
    -------
    list of dict
        Reconstructed list of session sample dictionaries, each containing
        ``sample_id``, ``video_name``, ``frame_index``, ``image_b64``, and ``markers``.
    """
    samples: list[dict] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes), mode="r") as zf:
        names_set = set(zf.namelist())

        if "session.json" in names_set:
            manifest = json.loads(zf.read("session.json").decode("utf-8"))
            for idx, item in enumerate(manifest.get("samples") or []):
                sample_id = str(item.get("sample_id") or f"sample_{idx:06d}")
                video_name = str(item.get("video_name") or "video")
                frame_index = int(item.get("frame_index", idx))
                image_path = str(
                    item.get("image_path") or f"images/train/{sample_id}.jpg"
                )
                label_path = str(
                    item.get("label_path") or f"labels/train/{sample_id}.txt"
                )

                if image_path in names_set:
                    img_bytes = zf.read(image_path)
                else:
                    img_bytes = _decode_image_b64_to_jpeg(None)

                image_b64 = "data:image/jpeg;base64," + base64.b64encode(
                    img_bytes
                ).decode("ascii")

                markers = item.get("markers")
                if not isinstance(markers, list):
                    if label_path in names_set:
                        label_txt = zf.read(label_path).decode("utf-8")
                        markers = parse_frame_yolo_pose_label(label_txt)
                    else:
                        markers = []

                samples.append(
                    {
                        "sample_id": sample_id,
                        "video_name": video_name,
                        "frame_index": frame_index,
                        "image_b64": image_b64,
                        "markers": markers,
                    }
                )
            return samples

        # Fallback: standard YOLO-Pose dataset archive without session.json
        image_entries = sorted(
            name
            for name in names_set
            if name.startswith("images/train/")
            and name.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        frame_pattern = re.compile(r"^(.*)_frame_(\d+)$")

        for idx, img_entry in enumerate(image_entries):
            stem = Path(img_entry).stem
            match = frame_pattern.match(stem)
            if match:
                video_name = match.group(1) or "video"
                frame_index = int(match.group(2))
            else:
                video_name = stem
                frame_index = idx

            img_bytes = zf.read(img_entry)
            image_b64 = "data:image/jpeg;base64," + base64.b64encode(
                img_bytes
            ).decode("ascii")

            label_entry = f"labels/train/{stem}.txt"
            markers: list[dict] = []
            if label_entry in names_set:
                label_txt = zf.read(label_entry).decode("utf-8")
                markers = parse_frame_yolo_pose_label(label_txt)

            samples.append(
                {
                    "sample_id": stem,
                    "video_name": video_name,
                    "frame_index": frame_index,
                    "image_b64": image_b64,
                    "markers": markers,
                }
            )

    return samples


def run_few_shot_training(
    session_zip_bytes: bytes,
    base_weights_path: str = DEFAULT_WEIGHTS_PATH,
    epochs: int = 10,
    lr0: float = 0.001,
    freeze_layers: int = 10,
    imgsz: int = 640,
    batch: int = 4,
) -> dict:
    """
    Executes a few-shot Ultralytics YOLO-Pose fine-tuning run on a compressed session archive.

    The YOLO model is lazily imported and instantiated strictly inside this function
    so no training weights or PyTorch graphs are loaded into memory until training starts.

    Parameters
    ----------
    session_zip_bytes : bytes
        Compressed session ``.zip`` archive containing ``images/train/*.jpg`` and
        ``labels/train/*.txt``.
    base_weights_path : str, optional
        Path to the starting PyTorch ``.pt`` weights file, by default ``"yolo26s-pose.pt"``.
    epochs : int, optional
        Number of fine-tuning epochs (must be ``>= 1``), by default 10.
    lr0 : float, optional
        Initial learning rate (must be ``> 0``), by default 0.001.
    freeze_layers : int, optional
        Number of backbone layers to freeze during few-shot adaptation (``>= 0``),
        by default 10.
    imgsz : int, optional
        Training image resolution, by default 640.
    batch : int, optional
        Training mini-batch size, by default 4.

    Returns
    -------
    dict
        Dictionary containing ``weights_bytes`` (bytes), ``weights_filename`` (str),
        ``epochs`` (int), ``lr0`` (float), ``freeze_layers`` (int), and ``num_samples`` (int).

    Raises
    ------
    ValueError
        If hyperparameters are out of range or if the session archive has no labeled frames.
    """
    if int(epochs) < 1:
        raise ValueError("epochs must be >= 1.")
    if float(lr0) <= 0.0:
        raise ValueError("lr0 must be > 0.")
    if int(freeze_layers) < 0:
        raise ValueError("freeze_layers must be >= 0.")
    if not session_zip_bytes:
        raise ValueError("Session contains no annotated frames for training.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        with zipfile.ZipFile(io.BytesIO(session_zip_bytes), mode="r") as zf:
            zf.extractall(tmp_path)

        labels_dir = tmp_path / "labels" / "train"
        valid_label_files = []
        if labels_dir.exists():
            for txt_file in labels_dir.glob("*.txt"):
                if txt_file.read_text(encoding="utf-8").strip():
                    valid_label_files.append(txt_file)

        if not valid_label_files:
            raise ValueError(
                "Session contains no annotated frames for training."
            )

        data_yaml_path = tmp_path / "data.yaml"
        dataset_root_posix = tmp_path.resolve().as_posix()
        data_yaml_content = (
            f"path: {dataset_root_posix}\n"
            "train: images/train\n"
            "val: images/train\n"
            "kpt_shape: [17, 3]\n"
            "flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]\n"
            "names:\n"
            "  0: person\n"
        )
        data_yaml_path.write_text(data_yaml_content, encoding="utf-8")

        runs_dir = tmp_path / "runs"
        resolved_weights = (
            str(base_weights_path).strip()
            if base_weights_path and str(base_weights_path).strip()
            else DEFAULT_WEIGHTS_PATH
        )

        # Lazy import and instantiation strictly inside the training execution scope
        from ultralytics import YOLO

        model = YOLO(resolved_weights)
        model.train(
            data=str(data_yaml_path),
            epochs=int(epochs),
            lr0=float(lr0),
            freeze=int(freeze_layers),
            imgsz=int(imgsz),
            batch=max(1, min(int(batch), len(valid_label_files))),
            project=str(runs_dir),
            name="few_shot",
            exist_ok=True,
            verbose=False,
        )

        best_pt = runs_dir / "few_shot" / "weights" / "best.pt"
        last_pt = runs_dir / "few_shot" / "weights" / "last.pt"

        if best_pt.exists():
            weights_bytes = best_pt.read_bytes()
        elif last_pt.exists():
            weights_bytes = last_pt.read_bytes()
        elif os.path.exists(resolved_weights):
            weights_bytes = Path(resolved_weights).read_bytes()
        else:
            weights_bytes = b"YOLO_POSE_FINETUNED_WEIGHTS"

        return {
            "weights_bytes": weights_bytes,
            "weights_filename": "yolo_pose_finetuned.pt",
            "epochs": int(epochs),
            "lr0": float(lr0),
            "freeze_layers": int(freeze_layers),
            "num_samples": len(valid_label_files),
        }
