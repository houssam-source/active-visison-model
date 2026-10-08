
import csv
import os
import threading
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from yolov3_backbone import YOLOv3TinyPerception, resolve_detector_model_path


PROJECT_ROOT = Path(__file__).resolve().parent
LATEST_TRAINED_MODEL_PATH = PROJECT_ROOT / "models" / "drone_yolov8n_30ep.pt"
BASELINE_MODEL_PATH = PROJECT_ROOT / "models" / "drone_yolov8n.pt"
BASE_MODEL_PATH = PROJECT_ROOT / "yolov8n.pt"
WEB_UI_PATH = PROJECT_ROOT / "static" / "index.html"
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"
MODEL_PATH = str(
    resolve_detector_model_path(
        default_paths=[
            LATEST_TRAINED_MODEL_PATH,
            BASELINE_MODEL_PATH,
            BASE_MODEL_PATH,
        ]
    )
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
cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"] ,
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


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@app.get("/", include_in_schema=False)
def dashboard():
    if not WEB_UI_PATH.is_file():
        raise HTTPException(status_code=404, detail="Perception dashboard is not installed.")
    return FileResponse(WEB_UI_PATH, media_type="text/html")


@app.get("/metrics")
def training_metrics():
    result_files = list(RUNS_DIR.glob("*/results.csv")) if RUNS_DIR.is_dir() else []
    result_files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    if not result_files:
        return {
            "available": False,
            "run": None,
            "summary": None,
            "history": [],
            "classes": [],
        }

    results_path = result_files[0]
    with results_path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    def metric_row(row):
        return {
            "epoch": _number(row.get("epoch")),
            "precision": _number(row.get("metrics/precision(B)")),
            "recall": _number(row.get("metrics/recall(B)")),
            "map50": _number(row.get("metrics/mAP50(B)")),
            "map5095": _number(row.get("metrics/mAP50-95(B)")),
            "train_box_loss": _number(row.get("train/box_loss")),
            "train_cls_loss": _number(row.get("train/cls_loss")),
            "train_dfl_loss": _number(row.get("train/dfl_loss")),
            "val_box_loss": _number(row.get("val/box_loss")),
            "val_cls_loss": _number(row.get("val/cls_loss")),
            "val_dfl_loss": _number(row.get("val/dfl_loss")),
        }

    history = [metric_row(row) for row in rows]
    class_reports = list(RUNS_DIR.glob("*_analysis/class_recall_metrics.csv"))
    class_reports.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    classes = []
    if class_reports:
        with class_reports[0].open("r", newline="", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
                classes.append({
                    "class_id": _number(row.get("class_id")),
                    "class_name": row.get("class_name", ""),
                    "instances": _number(row.get("ground_truth")),
                    "precision": _number(row.get("precision")),
                    "recall": _number(row.get("recall")),
                    "missed": _number(row.get("missed_as_background")),
                    "wrong_class": _number(row.get("wrong_class")),
                })

    return {
        "available": bool(history),
        "run": results_path.parent.name,
        "summary": history[-1] if history else None,
        "history": history,
        "classes": classes,
    }


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
        "checkpoint_exists": Path(MODEL_PATH).is_file(),
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
