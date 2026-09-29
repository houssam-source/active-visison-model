import os
import threading
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from yolov3_backbone import YOLOv3TinyPerception


PROJECT_ROOT = Path(__file__).resolve().parent
TRAINED_MODEL_PATH = PROJECT_ROOT / "models" / "drone_yolov8n.pt"
BASE_MODEL_PATH = PROJECT_ROOT / "yolov8n.pt"
MODEL_PATH = os.getenv(
    "YOLO_MODEL_PATH",
    str(TRAINED_MODEL_PATH if TRAINED_MODEL_PATH.is_file() else BASE_MODEL_PATH),
)
MODEL_WIDTH = 1920
MODEL_HEIGHT = 1080
LEARNING_EPOCHS = 10
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000

app = FastAPI(
    title="Embodied AI Perception API",
    description="YOLO object detection endpoints for the active-vision model.",
    version="1.0.0",
)

_detector = None
_model_lock = threading.Lock()
_inference_lock = threading.Lock()


class Detection(BaseModel):
    bbox: List[float]
    class_id: int
    class_name: str
    confidence: float
    center: List[float]


class DetectResponse(BaseModel):
    filename: str
    width: int
    height: int
    count: int
    detections: List[Detection]


def get_detector():
    global _detector
    if _detector is None:
        with _model_lock:
            if _detector is None:
                _detector = YOLOv3TinyPerception(
                    model_path=MODEL_PATH,
                    res=(MODEL_WIDTH, MODEL_HEIGHT),
                    epochs=LEARNING_EPOCHS,
                )
    return _detector


def _detect(image_rgb, confidence):
    with _inference_lock:
        detector = get_detector()
        return detector.detect_bounding_boxes(image_rgb, conf_thresh=confidence)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _detector is not None,
    }


@app.get("/model")
def model_info():
    detector = _detector
    names = getattr(detector.model, "names", {}) if detector is not None else {}
    if isinstance(names, dict):
        classes = [names[class_id] for class_id in sorted(names)]
    else:
        classes = []

    return {
        "checkpoint": MODEL_PATH,
        "loaded": detector is not None,
        "input_resolution": {"width": MODEL_WIDTH, "height": MODEL_HEIGHT},
        "learning_epochs": LEARNING_EPOCHS,
        "classes": classes,
    }


@app.post("/detect", response_model=DetectResponse)
async def detect(
    file: UploadFile = File(...),
    confidence: float = Query(default=0.25, ge=0.0, le=1.0),
):
    image_bytes = await file.read(MAX_IMAGE_BYTES + 1)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 20 MiB upload limit.")

    encoded_image = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=415, detail="Upload a valid encoded image.")

    height, width = image_bgr.shape[:2]
    if width * height > MAX_IMAGE_PIXELS:
        raise HTTPException(status_code=413, detail="Image dimensions exceed the 40-megapixel limit.")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    detections = await run_in_threadpool(_detect, image_rgb, confidence)
    return DetectResponse(
        filename=file.filename or "upload",
        width=width,
        height=height,
        count=len(detections),
        detections=detections,
    )
