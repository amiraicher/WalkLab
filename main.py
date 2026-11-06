"""
Main application entry point and FastAPI HTTP routes for WalkingLab.
"""
import shutil
import tempfile

from fastapi import File, UploadFile
from nicegui import app, run

import walkinglab.backend.inference as inference
from walkinglab.ui.app import run_app


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """
    Streams an uploaded video file to a temporary file on disk, runs batched YOLO-Pose
    inference inside the ``NamedTemporaryFile`` context manager, and automatically deletes
    the file upon exiting the ``with`` block.

    Parameters
    ----------
    file : fastapi.UploadFile
        The uploaded video file stream.

    Returns
    -------
    list of list of dict
        Per-frame lists of extracted keypoint marker dictionaries.
    """
    with tempfile.NamedTemporaryFile(
        suffix=".mp4", delete_on_close=False
    ) as temp_video:
        shutil.copyfileobj(file.file, temp_video)
        temp_video.close()
        return await run.cpu_bound(
            inference.process_video_file, {"path": temp_video.name}
        )


@app.get("/api/estimate")
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
        A dictionary containing the ``"estimated_seconds"`` key.
    """
    single_input_time = inference.get_empirical_time()
    batch_size = inference.BATCH_SIZE

    # Formula: single input inference time * number of frames / batch size
    estimated_seconds = single_input_time * (frames / batch_size)
    return {"estimated_seconds": estimated_seconds}


if __name__ in {"__main__", "__mp_main__"}:
    run_app()
