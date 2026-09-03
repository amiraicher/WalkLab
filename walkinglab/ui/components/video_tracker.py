from nicegui.element import Element
from typing import Callable, Any

class VideoTracker(Element, component='video_tracker.js'):
    def __init__(self, 
                 on_frame: Callable = None, 
                 on_file_uploaded_start: Callable = None,
                 on_file_uploaded_success: Callable = None) -> None:
        """
        A custom NiceGUI component for video tracking (Webcam & File Upload).
        
        Args:
            on_frame: Callback triggered when a new webcam frame is ready. Receives base64 encoded image string.
            on_file_uploaded_start: Callback triggered when a file upload starts.
            on_file_uploaded_success: Callback triggered when a file upload succeeds. Receives a dictionary with file info.
        """
        super().__init__()
        
        # Attach event listeners to bridge frontend events to Python callbacks
        if on_frame:
            self.on('frame_ready', lambda e: on_frame(e.args))
            
        if on_file_uploaded_start:
            self.on('file_uploaded_start', lambda e: on_file_uploaded_start(e.args))
            
        if on_file_uploaded_success:
            self.on('file_uploaded_success', lambda e: on_file_uploaded_success(e.args))
