import pytest
import numpy as np
from walkinglab.backend.inference import get_estimator, PoseEstimator

def test_estimator_initialization():
    """Test that the OpenVINO PoseEstimator initializes and compiles correctly."""
    est = get_estimator()
    assert est is not None
    assert est.compiled_model is not None
    assert est.infer_request is not None

def test_estimator_inference():
    """Test that the estimator can perform inference on a dummy tensor without crashing."""
    est = get_estimator()
    # Create a dummy image (e.g., solid black 640x640)
    dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Process the frame
    markers = est.infer(dummy_image)
    
    # It should return a list (even if empty, though a completely black image 
    # probably won't have confidence > 0.5 for a person)
    assert isinstance(markers, list)

