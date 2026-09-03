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

if __name__ in {"__main__", "__mp_main__"}:
    run_app()
