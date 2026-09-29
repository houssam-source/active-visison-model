import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = Path.home() / "Downloads" / "yolo_dataset" / "data.yaml"
DEFAULT_MODEL = PROJECT_ROOT / "yolov8n.pt"
OUTPUT_MODEL = PROJECT_ROOT / "models" / "drone_yolov8n.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLO detector on the prepared dataset.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to the dataset data.yaml.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Pretrained YOLO checkpoint or model name.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", default="auto", help="auto, cpu, or a CUDA device such as 0.")
    parser.add_argument("--workers", type=int, default=0, help="Data-loader workers; 0 is Windows-friendly.")
    parser.add_argument("--name", default="drone_train", help="Ultralytics run name.")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = args.data.expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")
    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1:
        raise ValueError("epochs, imgsz, and batch must be positive (imgsz at least 32).")

    device = args.device
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name=args.name,
    )

    best_checkpoint = Path(model.trainer.best)
    if not best_checkpoint.is_file():
        raise FileNotFoundError(f"Training finished without a best checkpoint: {best_checkpoint}")
    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_checkpoint, OUTPUT_MODEL)
    print(f"Best checkpoint: {best_checkpoint}")
    print(f"Inference checkpoint: {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()