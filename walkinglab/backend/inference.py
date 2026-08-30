"""
Inference module for YOLO-Pose tracking.
"""

def extract_keypoints(video_source):
    pass

def process_frame(frame_data: str):
    """Placeholder for real-time webcam frame processing."""
    # frame_data is base64 encoded image
    # return dummy coordinates representing the keypoints for the frame
    return [{"x": 100, "y": 100}]

def process_video_file(file_info: dict):
    """Placeholder for offline batch processing of a video file."""
    # Trigger an async task to process the video and return track data
    return {"status": "processing", "file": file_info.get("name")}
