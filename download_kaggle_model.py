import argparse
import shutil
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_HANDLE = "dark77/active-vision-model/pytorch/default"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "drone_yolov8n_30ep.pt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a Kaggle Model checkpoint into this project."
    )
    parser.add_argument(
        "--handle",
        default=DEFAULT_HANDLE,
        help="Kaggle model handle: owner/model/framework/variation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Local destination for the downloaded .pt checkpoint.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "KaggleHub is required. Install it with: "
            "python -m pip install kagglehub"
        ) from exc

    with tempfile.TemporaryDirectory(prefix="kaggle-yolo-") as temp_dir:
        downloaded_dir = Path(
            kagglehub.model_download(args.handle, output_dir=temp_dir)
        )
        checkpoints = sorted(downloaded_dir.rglob("*.pt"))
        preferred_names = (
            "drone_yolov8n_30ep.pt",
            "drone_yolov8n.pt",
            "best.pt",
        )
        checkpoint = next(
            (path for name in preferred_names for path in checkpoints if path.name == name),
            None,
        )
        if checkpoint is None and len(checkpoints) == 1:
            checkpoint = checkpoints[0]
        if checkpoint is None:
            raise FileNotFoundError(
                "Could not select one YOLO checkpoint from the downloaded model. "
                f"Found: {[str(path.relative_to(downloaded_dir)) for path in checkpoints]}"
            )

        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint, output_path)

    print(f"Downloaded Kaggle model: {args.handle}")
    print(f"Local checkpoint: {output_path}")


if __name__ == "__main__":
    main()