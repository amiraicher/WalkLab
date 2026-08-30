from nicegui.element import Element
from typing import Callable, Any

class VideoTracker(Element, component='video_tracker.js'):
    def __init__(self, on_frame: Callable[[str], Any] = None, on_upload: Callable[[dict], Any] = None) -> None:
        """
        A custom NiceGUI component for video tracking (Webcam & File Upload).
        
        Args:
            on_frame: Callback triggered when a new webcam frame is ready. Receives base64 encoded image string.
            on_upload: Callback triggered when a file is uploaded. Receives a dictionary with file info.
        """
        super().__init__()
        
        # Attach event listeners to bridge frontend events to Python callbacks
        if on_frame:
            self.on('frame_ready', lambda e: on_frame(e.args))
            
        if on_upload:
            self.on('file_uploaded', lambda e: on_upload(e.args))
