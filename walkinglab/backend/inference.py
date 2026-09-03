"""
Inference module for YOLO-Pose tracking using OpenVINO native runtime.
"""
import cv2
import numpy as np
import openvino as ov
import base64
import time
import logging
import os
import tempfile
from ultralytics.utils.ops import scale_coords
from ultralytics import YOLO

import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

class PoseEstimator:
    def __init__(self, model_path: str):
        """
        Initializes the native OpenVINO runtime session for YOLO pose estimation.

        Parameters
        ----------
        model_path : str
            Path to the YOLO PyTorch (.pt) or OpenVINO (.xml) model file.
        """
        if model_path.endswith('.pt'):
            ov_dir = model_path.replace('.pt', '_openvino_model')
            ov_xml = os.path.join(ov_dir, model_path.replace('.pt', '.xml'))
            if not os.path.exists(ov_xml):
                logger.info(f"OpenVINO model not found. Exporting {model_path}...")
                model = YOLO(model_path)
                model.export(format='openvino')
            model_path = ov_xml
            
        # Initialize native OpenVINO Core
        core = ov.Core()
        
        # CPU is used here. OpenVINO will automatically optimize for Intel CPUs.
        self.compiled_model = core.compile_model(model_path, "CPU")
        self.infer_request = self.compiled_model.create_infer_request()
        
        self.input_layer = self.compiled_model.input(0)
        
        # Typically shape is [1, 3, 640, 640]
        self.input_shape = self.input_layer.shape
        self.img_size = (self.input_shape[2], self.input_shape[3])

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, tuple]:
        """
        Manually prepares the image for YOLO inference.

        Parameters
        ----------
        image : numpy.ndarray
            The original BGR image loaded via OpenCV.

        Returns
        -------
        tuple
            (preprocessed_image_tensor, scale_ratio, padding_tuple)
        """
        height, width = image.shape[:2]
        scale_ratio = min(self.img_size[0] / height, self.img_size[1] / width)
        unpadded_dimensions = int(round(width * scale_ratio)), int(round(height * scale_ratio))
        pad_w, pad_h = self.img_size[1] - unpadded_dimensions[0], self.img_size[0] - unpadded_dimensions[1]
        pad_w /= 2  # divide padding into 2 sides
        pad_h /= 2

        if (width, height) != unpadded_dimensions:
            img = cv2.resize(image, unpadded_dimensions, interpolation=cv2.INTER_LINEAR)
        else:
            img = image.copy()

        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # Convert BGR to RGB, HWC to CHW
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        # Normalize and add batch dimension
        img_tensor = img.astype(np.float32) / 255.0
        img_tensor = np.expand_dims(img_tensor, axis=0)

        return img_tensor, scale_ratio, (pad_w, pad_h)

    def postprocess(self, output: np.ndarray, orig_img_shape: tuple) -> list[dict]:
        """
        Extracts End-to-End predicted keypoints and maps coordinates 
        back to the original image dimensions.

        Parameters
        ----------
        output : numpy.ndarray
            The raw output tensor from the OpenVINO model. Expected shape: [1, 300, 57]
        orig_img_shape : tuple
            The shape of the original image (H, W, C).

        Returns
        -------
        list of dict
            A list containing the keypoints normalized [0, 1] relative to the original image.
        """
        # output is [1, 300, 57] where 57 = 4(bbox) + 1(conf) + 1(cls) + 51(17 keypoints * 3)
        preds = output[0] # shape [300, 57]
        
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
        
        # scale_coords scales the coords inplace or returns scaled coords
        import torch
        scaled_kpts = scale_coords(self.img_size, torch.from_numpy(kpts), orig_img_shape)
        
        markers = []
        h, w = orig_img_shape[:2]
        for kp in scaled_kpts[0]:
            x, y, v = kp.tolist()
            if v > 0.5:
                markers.append({"x": x / w, "y": y / h})
                
        return markers

    def infer(self, image: np.ndarray) -> list[dict]:
        """
        Helper to run the full inference pipeline (preprocess, infer, postprocess) on an image array.
        
        Parameters
        ----------
        image : numpy.ndarray
            The BGR image array.
            
        Returns
        -------
        list of dict
            List of marker coordinates.
        """
        img_tensor, scale_ratio, pad = self.preprocess(image)
        
        results = self.infer_request.infer([img_tensor])
        output = next(iter(results.values()))
        
        markers = self.postprocess(output, image.shape)
        return markers

    def predict(self, frame_data_b64: str) -> list[dict]:
        """
        Main pipeline to be called by the frontend via WebSockets.

        Parameters
        ----------
        frame_data_b64 : str
            Base64 encoded image string (data:image/jpeg;base64,...).

        Returns
        -------
        list of dict
            List of marker coordinates.
        """
        t_start = time.perf_counter()
        
        # 1. Decode base64 to numpy array
        header, encoded = frame_data_b64.split(",", 1)
        data = base64.b64decode(encoded)
        np_arr = np.frombuffer(data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        t_pre = time.perf_counter()
        # 2. Preprocess
        img_tensor, scale_ratio, pad = self.preprocess(image)
        
        t_inf = time.perf_counter()
        # 3. Infer
        results = self.infer_request.infer([img_tensor])
        output = next(iter(results.values()))
        
        t_post = time.perf_counter()
        # 4. Postprocess
        markers = self.postprocess(output, image.shape)
        
        t_end = time.perf_counter()
        
        logger.debug(f"Timing - Preprocess: {(t_pre-t_start)*1000:.1f}ms, Inference: {(t_inf-t_pre)*1000:.1f}ms, Postprocess: {(t_end-t_post)*1000:.1f}ms, Total: {(t_end-t_start)*1000:.1f}ms")
        
        return markers

# Global singleton for the estimator
estimator = None

def get_estimator():
    """
    Retrieves the global PoseEstimator singleton, initializing it if necessary.

    Returns
    -------
    PoseEstimator
        The initialized pose estimator instance.
    """
    global estimator
    if estimator is None:
        estimator = PoseEstimator('yolo26s-pose.pt')
    return estimator

def process_frame(frame_data: str):
    """Processes a real-time webcam frame and returns marker positions."""
    return get_estimator().predict(frame_data)

def process_video_file(file_info: dict) -> list:
    """
    Processes a video file asynchronously using OpenVINO AsyncInferQueue.
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
    infer_queue = ov.AsyncInferQueue(est.compiled_model, 0)
    
    def callback(request, userdata):
        frame_idx, orig_shape = userdata
        output = next(iter(request.results.values()))
        markers = est.postprocess(output, orig_shape)
        all_markers[frame_idx] = markers
        
    infer_queue.set_callback(callback)
    
    frame_idx = 0
    t_start = time.perf_counter()
    logger.info(f"🚀 [Batch Mode] Starting Async inference on {video_path}...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        img_tensor, scale_ratio, pad = est.preprocess(frame)
        
        # Start inference concurrently (blocks only if the queue is full)
        infer_queue.start_async([img_tensor], userdata=(frame_idx, frame.shape))
        
        frame_idx += 1
        if frame_idx % 100 == 0:
            logger.info(f"[Batch Mode] Queued {frame_idx} frames...")
            
    # Wait for the remaining frames in the queue to finish processing
    infer_queue.wait_all()
    cap.release()
    
    t_end = time.perf_counter()
    if frame_idx > 0:
        fps = frame_idx / (t_end - t_start)
        logger.info(f"✅ [Batch Mode] Complete! Processed {frame_idx} frames in {t_end - t_start:.2f}s ({fps:.1f} FPS)")
        
    # Return only the frames we actually read
    return all_markers[:frame_idx]
