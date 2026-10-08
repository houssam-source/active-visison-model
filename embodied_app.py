import csv
import os
import threading
from pathlib import Path
from typing import List

import cv2
import numpy as np
import base64

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from yolov3_backbone import YOLOv3TinyPerception, resolve_detector_model_path

try:
    from swarmmanager import EmbodiedSwarmRuntime
except Exception:  # pragma: no cover - simulator stack is optional
    EmbodiedSwarmRuntime = None

try:
    from genesis_bridge import GenesisSwarmBridge
except Exception:  # pragma: no cover - simulator is optional
    GenesisSwarmBridge = None


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    PROJECT_ROOT / "models" / "drone_yolov8n_30ep.pt",
    PROJECT_ROOT / "models" / "drone_yolov8n.pt",
    PROJECT_ROOT / "yolov8n.pt",
]

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000


class EmbodiedRuntime:
    def __init__(self, detector_model_path=None, resolution=(640, 480), fov_deg=45.0):
        self.model_path = resolve_detector_model_path(
            model_path=detector_model_path,
            default_paths=MODEL_CANDIDATES,
            env_var="YOLO_MODEL_PATH",
        )
        self.resolution = tuple(resolution)
        self.fov_deg = float(fov_deg)
        self.detector = None
        self.simulation = None

    def get_detector(self):
        if self.detector is None:
            self.detector = YOLOv3TinyPerception(
                model_path=str(self.model_path),
                res=self.resolution,
                fov_deg=self.fov_deg,
            )
        return self.detector

    def get_simulation(self, n_drones=1):
        if self.simulation is None:
            if EmbodiedSwarmRuntime is not None:
                self.simulation = EmbodiedSwarmRuntime(
                    n_drones=n_drones,
                    detector_model_path=str(self.model_path) if self.model_path.is_file() else None,
                    control_hz=10,
                    gui=False,
                    skip_simulation=False,
                    software_simulation=True,
                )
                return self.simulation
            if GenesisSwarmBridge is None:
                raise RuntimeError("Genesis is not installed; the simulator feature is unavailable.")
            self.simulation = GenesisSwarmBridge(
                detector=self.get_detector(),
                detector_model_path=str(self.model_path),
                control_hz=10,
                show_viewer=False,
                camera_gui=False,
            )
        return self.simulation

    def detect(self, image_rgb, confidence):
        return self.get_detector().detect_bounding_boxes(image_rgb, conf_thresh=confidence)


runtime = EmbodiedRuntime()
_runtime_lock = threading.Lock()

app = FastAPI(
    title="Embodied AI Runtime",
    description="Unified app combining detection and simulation in one runtime.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "static")), name="static")

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000,http://127.0.0.1:8001,http://localhost:8001"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class Detection:
    def __init__(self, bbox, class_id, class_name, confidence, center):
        self.bbox = bbox
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.center = center


class DetectResponse:
    def __init__(self, filename, width, height, detections):
        self.filename = filename
        self.width = width
        self.height = height
        self.count = len(detections)
        self.detections = detections


@app.get("/", include_in_schema=False)
def dashboard():
    index_path = PROJECT_ROOT / "static" / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="Dashboard file not found.")
    return FileResponse(index_path)


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@app.get("/metrics")
def training_metrics():
    runs_dir = PROJECT_ROOT / "runs" / "detect"
    result_files = list(runs_dir.glob("*/results.csv")) if runs_dir.is_dir() else []
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
    class_reports = list(runs_dir.glob("*_analysis/class_recall_metrics.csv"))
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


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": runtime.detector is not None,
        "simulator_available": runtime.simulation is not None or EmbodiedSwarmRuntime is not None or GenesisSwarmBridge is not None,
        "checkpoint": str(runtime.model_path),
    }


@app.get("/model")
def model_info():
    detector = runtime.detector
    names = getattr(detector.model, "names", {}) if detector is not None else {}
    classes = [names[class_id] for class_id in sorted(names)] if isinstance(names, dict) else []
    return {
        "checkpoint": str(runtime.model_path),
        "checkpoint_exists": runtime.model_path.is_file(),
        "loaded": detector is not None,
        "input_resolution": {"width": runtime.resolution[0], "height": runtime.resolution[1]},
        "classes": classes,
        "simulator_available": GenesisSwarmBridge is not None,
    }


@app.post("/detect")
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
    with _runtime_lock:
        detections = runtime.detect(image_rgb, confidence)

    return {
        "filename": file.filename or "upload",
        "width": width,
        "height": height,
        "count": len(detections),
        "detections": detections,
    }


def _advance_simulation(sim, frames: int = 1):
    if not hasattr(sim, "step"):
        return getattr(sim, "summary", lambda: {})()
    control_steps = getattr(getattr(sim, "env", None), "control_steps", 1) or 1
    steps = max(1, int(frames), int(control_steps))
    return sim.step(steps)


@app.post("/simulation/start")
def start_simulation(frames: int = 1, camera_gui: bool = False):
    try:
        sim = runtime.get_simulation(n_drones=1)
    except RuntimeError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    try:
        payload = _advance_simulation(sim, frames)
        summary = payload if isinstance(payload, dict) else getattr(sim, "summary", lambda: {})()
    except Exception:
        summary = getattr(sim, "summary", lambda: {})()
    return {
        "started": True,
        "frames_requested": int(frames),
        "camera_gui": bool(camera_gui),
        "mode": "embodied-swarm-runtime",
        "checkpoint": str(runtime.model_path),
        "simulator_ready": sim is not None,
        "summary": summary,
    }


@app.get("/simulation/view")
def simulation_view():
    try:
        sim = runtime.get_simulation(n_drones=1)
    except RuntimeError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc

    preview = sim.preview() if hasattr(sim, "preview") else {"available": False, "message": "Preview unavailable"}
    if not preview.get("available"):
        _advance_simulation(sim, 1)
        preview = sim.preview() if hasattr(sim, "preview") else {"available": False, "message": "Preview unavailable"}

    state_json = "{}"
    if preview.get("agent_state") is not None:
        import json
        state_json = json.dumps(preview["agent_state"])

    frame_src = ""
    if preview.get("frame"):
        frame_src = f"data:image/jpeg;base64,{preview['frame']}"

    return f"""
    <html><head><title>Embodied Swarm Live View</title>
    <style>body{{font-family:Segoe UI,sans-serif;background:#111;color:#eee;display:grid;place-items:center;min-height:100vh;margin:0}} .panel{{max-width:900px;width:92vw;background:#1a1a1a;border:1px solid #333;border-radius:12px;padding:20px;box-shadow:0 8px 30px rgba(0,0,0,.35)}} img{{max-width:100%;border-radius:10px;border:1px solid #2f2f2f;display:block;margin:12px auto;background:#000}} .meta{{font-size:12px;color:#b9b9b9;margin-top:10px;white-space:pre-wrap;word-break:break-all}} h1{{margin:0 0 12px;font-size:22px}} </style></head>
    <body><div class='panel'><h1>Embodied Swarm Live View</h1>{f'<img src="{frame_src}" alt="Swarm camera preview" />' if frame_src else '<p>No frame available yet. Run the simulation and refresh this page.</p>'}<div class='meta'>State: {state_json}</div></div></body></html>
    """


@app.get("/simulation/preview")
def simulation_preview():
    try:
        sim = runtime.get_simulation(n_drones=1)
    except RuntimeError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    _advance_simulation(sim, 1)
    preview = sim.preview() if hasattr(sim, "preview") else {"available": False, "message": "Preview unavailable"}
    return preview


def main():
    import uvicorn

    uvicorn.run("embodied_app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
