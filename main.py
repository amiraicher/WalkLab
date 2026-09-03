from walkinglab.ui.app import run_app
from nicegui import app, run
from fastapi import UploadFile, File
import walkinglab.backend.inference as inference

import tempfile
import shutil
import os

@app.post('/api/upload_video')
async def upload_video(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        shutil.copyfileobj(file.file, temp_video)
        temp_video_path = temp_video.name
        
    try:
        markers = await run.cpu_bound(inference.process_video_file, {"path": temp_video_path})
    finally:
        os.remove(temp_video_path)
        
    return markers

@app.get('/api/estimate')
def estimate_time(frames: int):
    """
    Estimates the time required for batch inference based on empirical tracking.

    Parameters
    ----------
    frames : int
        The total number of frames in the uploaded video.

    Returns
    -------
    dict
        A dictionary containing the 'estimated_seconds' key.
    """
    single_input_time = inference.get_empirical_time()
    batch_size = inference.BATCH_SIZE
    
    # Formula: single input inference time * number of frames / batch size
    estimated_seconds = single_input_time * (frames / batch_size)
    return {"estimated_seconds": estimated_seconds}

if __name__ in {"__main__", "__mp_main__"}:
    run_app()
