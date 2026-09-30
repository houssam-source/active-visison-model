import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
KAGGLE_DATA = Path("/kaggle/working/yolo_data.yaml")
DEFAULT_DATA = (
    KAGGLE_DATA
    if KAGGLE_DATA.is_file()
    else Path.home() / "Downloads" / "yolo_dataset" / "data.yaml"
)
DEFAULT_MODEL = PROJECT_ROOT / "yolov8n.pt"
OUTPUT_MODEL = PROJECT_ROOT / "models" / "drone_yolov8n.pt"
DEFAULT_PROJECT = PROJECT_ROOT / "runs" / "detect"
RECALL_KEY = "metrics/recall(B)"


class RecallAwareEarlyStopping:
    def __init__(
        self,
        min_epochs=12,
        recall_patience=6,
        stability_window=5,
        recall_min_delta=0.002,
        loss_tolerance=0.03,
        metric_tolerance=0.01,
    ):
        self.min_epochs = min_epochs
        self.recall_patience = recall_patience
        self.stability_window = stability_window
        self.recall_min_delta = recall_min_delta
        self.loss_tolerance = loss_tolerance
        self.metric_tolerance = metric_tolerance
        self.best_recall = None
        self.epochs_without_recall_gain = 0
        self.history = []

    @staticmethod
    def _is_stable(records, section, key, tolerance, relative):
        values = [record[section][key] for record in records]
        spread = max(values) - min(values)
        scale = max(abs(sum(values) / len(values)), 1e-8) if relative else 1.0
        return spread <= tolerance * scale

    def __call__(self, trainer):
        metrics = trainer.metrics or {}
        if RECALL_KEY not in metrics:
            return

        loss_values = torch.as_tensor(trainer.tloss).detach().cpu().flatten().tolist()
        loss_names = list(getattr(trainer, "loss_names", []))
        if len(loss_names) != len(loss_values):
            loss_names = [f"loss_{index}" for index in range(len(loss_values))]

        train_losses = {
            name: float(value) for name, value in zip(loss_names, loss_values)
        }
        val_losses = {
            key: float(value)
            for key, value in metrics.items()
            if key.startswith("val/") and key.endswith("_loss")
        }
        val_scores = {
            key: float(value)
            for key, value in metrics.items()
            if key.startswith("metrics/") and key != RECALL_KEY
        }
        if not train_losses or not val_losses or not val_scores:
            return

        recall = float(metrics[RECALL_KEY])
        if self.best_recall is None or recall > self.best_recall + self.recall_min_delta:
            self.best_recall = recall
            self.epochs_without_recall_gain = 0
        else:
            self.epochs_without_recall_gain += 1

        self.history.append({
            "epoch": int(trainer.epoch) + 1,
            "recall": recall,
            "best_recall": self.best_recall,
            "epochs_without_recall_gain": self.epochs_without_recall_gain,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_scores": val_scores,
        })

        if int(trainer.epoch) + 1 < self.min_epochs:
            return
        if self.epochs_without_recall_gain < self.recall_patience:
            return
        if len(self.history) < self.stability_window:
            return

        window = self.history[-self.stability_window:]
        stable = all(
            self._is_stable(window, section, key, self.loss_tolerance, True)
            for section in ("train_losses", "val_losses")
            for key in window[-1][section]
        ) and all(
            self._is_stable(window, "val_scores", key, self.metric_tolerance, False)
            for key in window[-1]["val_scores"]
        )

        if stable:
            print(
                "Early stopping: recall has not improved by "
                f"{self.recall_min_delta:.3f} for {self.recall_patience} epochs, "
                "and train/validation metrics are stable."
            )
            trainer.stop = True


def save_training_history(path, history):
    if not history:
        return

    train_keys = sorted(history[0]["train_losses"])
    val_loss_keys = sorted(history[0]["val_losses"])
    val_score_keys = sorted(history[0]["val_scores"])
    fieldnames = [
        "epoch",
        "recall",
        "best_recall",
        "epochs_without_recall_gain",
        *(f"train/{key}" for key in train_keys),
        *val_loss_keys,
        *val_score_keys,
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in history:
            row = {
                key: record[key]
                for key in ("epoch", "recall", "best_recall", "epochs_without_recall_gain")
            }
            row.update({f"train/{key}": value for key, value in record["train_losses"].items()})
            row.update(record["val_losses"])
            row.update(record["val_scores"])
            writer.writerow(row)


def save_confusion_report(model_path, data_path, project_dir, args, device):
    model = YOLO(str(model_path))
    validation = model.val(
        data=str(data_path),
        imgsz=args.imgsz,
        batch=args.batch,
        split="val",
        device=device,
        workers=args.workers,
        plots=True,
        project=str(project_dir),
        name=f"{args.name}_analysis",
        exist_ok=True,
    )

    class_names = [model.names[index] for index in range(len(model.names))]
    class_count = len(class_names)
    matrix = np.rint(validation.confusion_matrix.matrix).astype(np.int64)
    if matrix.shape != (class_count + 1, class_count + 1):
        raise ValueError(f"Unexpected confusion matrix shape: {matrix.shape}")

    report_dir = project_dir / f"{args.name}_analysis"
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["predicted\\actual", *class_names, "background"])
        for index, row in enumerate(matrix):
            label = class_names[index] if index < class_count else "background"
            writer.writerow([label, *row.tolist()])

    class_rows = []
    for class_id, class_name in enumerate(class_names):
        true_positives = int(matrix[class_id, class_id])
        ground_truth = int(matrix[:, class_id].sum())
        predicted = int(matrix[class_id, :].sum())
        missed_as_background = int(matrix[class_count, class_id])
        wrong_class = ground_truth - true_positives - missed_as_background
        class_rows.append({
            "class_id": class_id,
            "class_name": class_name,
            "ground_truth": ground_truth,
            "true_positive": true_positives,
            "missed_as_background": missed_as_background,
            "wrong_class": wrong_class,
            "false_positive": predicted - true_positives,
            "precision": true_positives / predicted if predicted else 0.0,
            "recall": true_positives / ground_truth if ground_truth else 0.0,
        })

    with (report_dir / "class_recall_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(class_rows[0]))
        writer.writeheader()
        writer.writerows(class_rows)

    total_ground_truth = sum(row["ground_truth"] for row in class_rows)
    total_missed = sum(row["missed_as_background"] for row in class_rows)
    total_wrong_class = sum(row["wrong_class"] for row in class_rows)
    dominant_error = (
        "missed detections (ground truth assigned to background)"
        if total_missed > total_wrong_class
        else "class confusion"
    )
    analysis = [
        "Validation confusion-matrix analysis",
        "Rows are predicted classes; columns are ground-truth classes. The final row/column is background.",
        f"Ground-truth objects: {total_ground_truth}",
        f"Missed as background: {total_missed} ({total_missed / max(total_ground_truth, 1):.1%})",
        f"Assigned to a wrong class: {total_wrong_class}",
        f"Dominant error pattern: {dominant_error}.",
        "Class recall from this confusion matrix:",
    ]
    analysis.extend(
        f"- {row['class_name']}: {row['recall']:.3f} "
        f"({row['missed_as_background']} missed, {row['wrong_class']} wrong class)"
        for row in sorted(class_rows, key=lambda item: item["recall"])
    )
    analysis.extend([
        "Interpretation: if background misses dominate, the main issue is object proposal/localization or confidence acceptance, not confusing the five classes.",
        "Inspect small, blurred, occluded, and poorly labeled examples; also compare confidence thresholds before changing class names or adding class weights.",
    ])
    (report_dir / "recall_analysis.txt").write_text("\n".join(analysis) + "\n", encoding="utf-8")
    print(f"Confusion matrix and recall analysis saved to: {report_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train and analyze the custom YOLO detector.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to the dataset data.yaml.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Pretrained YOLO checkpoint or model name.")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT, help="Directory for training and analysis outputs.")
    parser.add_argument("--output-model", type=Path, default=OUTPUT_MODEL, help="Path for the exported best checkpoint.")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", default="auto", help="auto, cpu, or a CUDA device such as 0.")
    parser.add_argument("--workers", type=int, default=0, help="Data-loader workers; 0 is Windows-friendly.")
    parser.add_argument("--name", default="drone_train_30ep", help="Ultralytics run name.")
    parser.add_argument("--min-epochs", type=int, default=12, help="Minimum epochs before custom early stopping.")
    parser.add_argument("--recall-patience", type=int, default=6, help="Stop-check patience without recall gain.")
    parser.add_argument("--stability-window", type=int, default=5, help="Recent epochs used to assess metric stability.")
    parser.add_argument("--recall-min-delta", type=float, default=0.002, help="Minimum recall improvement to reset patience.")
    parser.add_argument("--analyze-only", action="store_true", help="Generate confusion and recall reports without training.")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = args.data.expanduser().resolve()
    project_dir = args.project.expanduser().resolve()
    output_model = args.output_model.expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")
    if args.imgsz < 32 or args.batch < 1:
        raise ValueError("imgsz must be at least 32 and batch must be positive.")
    if args.analyze_only:
        model_path = Path(args.model).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Analysis checkpoint not found: {model_path}")
    elif (
        args.epochs < 1
        or args.min_epochs < 1
        or args.min_epochs > args.epochs
        or args.recall_patience < 1
        or args.stability_window < 2
        or args.recall_min_delta < 0
    ):
        raise ValueError("Invalid epoch, early-stopping, or recall threshold settings.")

    device = args.device
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"

    if args.analyze_only:
        save_confusion_report(model_path, data_path, project_dir, args, device)
        return

    model = YOLO(args.model)
    early_stopper = RecallAwareEarlyStopping(
        min_epochs=args.min_epochs,
        recall_patience=args.recall_patience,
        stability_window=args.stability_window,
        recall_min_delta=args.recall_min_delta,
    )
    model.add_callback("on_fit_epoch_end", early_stopper)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project=str(project_dir),
        name=args.name,
        patience=args.epochs,
        plots=True,
    )

    best_checkpoint = Path(model.trainer.best)
    if not best_checkpoint.is_file():
        raise FileNotFoundError(f"Training finished without a best checkpoint: {best_checkpoint}")
    output_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_checkpoint, output_model)
    run_dir = Path(model.trainer.save_dir)
    save_training_history(run_dir / "early_stopping_metrics.csv", early_stopper.history)
    print(f"Best checkpoint: {best_checkpoint}")
    print(f"Inference checkpoint: {output_model}")
    save_confusion_report(output_model, data_path, project_dir, args, device)


if __name__ == "__main__":
    main()