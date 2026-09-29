import os
import numpy as np
import torch
import torch.nn.functional as F

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - handled at runtime for optional environments
    YOLO = None


class YOLOv3TinyPerception:
    def __init__(self, model_path="yolov8n.pt", conf_thresh=0.25, fov_deg=90.0, res=(1920, 1080), epochs=10):
        if YOLO is None:
            raise ImportError(
                "ultralytics is not installed in the active Python environment. "
                "Install it with: pip install ultralytics"
            )
        # Keep the lightweight YOLO family configurable while using the correct Ultralytics API.
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.fov_deg = float(fov_deg)
        self.W, self.H = res
        self.epochs = int(epochs)
        self.model_h, self.model_w = self._next_valid_size(self.H, self.W)
        self.imgsz = (self.model_h, self.model_w)
        fov_rad = np.deg2rad(fov_deg)
        self.fx = self.fy = self.W / (2 * np.tan(fov_rad / 2))
        self.cx, self.cy = self.W / 2, self.H / 2
        self._warmup()

    @staticmethod
    def _next_valid_size(height, width):
        new_h = int(np.ceil(height / 32.0) * 32)
        new_w = int(np.ceil(width / 32.0) * 32)
        return max(32, new_h), max(32, new_w)

    def _prepare_rgb_for_model(self, rgb):
        if rgb is None or np.size(rgb) == 0:
            return None, 1.0, 1.0

        rgb_array = np.ascontiguousarray(rgb)
        if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
            return None, 1.0, 1.0
        if rgb_array.dtype != np.uint8:
            if np.issubdtype(rgb_array.dtype, np.floating) and rgb_array.size and np.nanmax(rgb_array) <= 1.0:
                rgb_array = rgb_array * 255.0
            rgb_array = np.clip(rgb_array, 0, 255).astype(np.uint8)

        in_h, in_w = rgb_array.shape[:2]
        target_h, target_w = self._next_valid_size(in_h, in_w)

        tensor = torch.from_numpy(rgb_array).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        if in_h != target_h or in_w != target_w:
            tensor = F.interpolate(tensor, size=(target_h, target_w), mode="bilinear", align_corners=False)

        return tensor, float(target_w / in_w), float(target_h / in_h)

    def train(self, data, epochs=None, **kwargs):
        """Train the underlying YOLO model for the configured epoch count."""
        train_epochs = self.epochs if epochs is None else int(epochs)
        return self.model.train(data=data, epochs=train_epochs, imgsz=self.imgsz, **kwargs)

    def _warmup(self):
        dummy = torch.zeros(1, 3, self.model_h, self.model_w, device=self.model.device)
        self.model(dummy, verbose=False, conf=self.conf_thresh)

    def detect_bounding_boxes(self, rgb, conf_thresh=None):
        """Return the detected bounding boxes as a list of dictionaries."""
        if rgb is None or np.size(rgb) == 0:
            return []

        conf = self.conf_thresh if conf_thresh is None else conf_thresh
        rgb_tensor, scale_x, scale_y = self._prepare_rgb_for_model(rgb)
        if rgb_tensor is None:
            return []

        with torch.no_grad():
            results = self.model(rgb_tensor, verbose=False, conf=conf)

        if not results or len(results) == 0 or results[0].boxes is None:
            return []

        output = []
        names = getattr(self.model, "names", {}) or {}
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().tolist()
            xyxy = [
                xyxy[0] / scale_x,
                xyxy[1] / scale_y,
                xyxy[2] / scale_x,
                xyxy[3] / scale_y,
            ]
            cls_id = int(box.cls[0].item()) if box.cls is not None else -1
            conf_value = float(box.conf[0].item()) if box.conf is not None else 0.0
            label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)

            output.append({
                "bbox": [float(v) for v in xyxy],
                "class_id": cls_id,
                "class_name": label,
                "confidence": conf_value,
                "center": [
                    (xyxy[0] + xyxy[2]) / 2.0,
                    (xyxy[1] + xyxy[3]) / 2.0,
                ],
            })
        return output

    def _sample_depth(self, depth_map, u, v, patch_size=3):
        depth_h, depth_w = depth_map.shape[:2]
        y_min, y_max = max(0, int(v) - patch_size), min(depth_h, int(v) + patch_size)
        x_min, x_max = max(0, int(u) - patch_size), min(depth_w, int(u) + patch_size)
        patch = depth_map[y_min:y_max, x_min:x_max]
        valid = patch[np.isfinite(patch) & (patch > 0.05)]
        return float(np.median(valid)) if len(valid) > 0 else -1.0

    def __call__(self, rgb, depth):
        if rgb is None or depth is None or np.size(rgb) == 0 or np.size(depth) == 0:
            return []

        if torch.is_tensor(rgb):
            rgb = rgb.detach().cpu().numpy()
        if torch.is_tensor(depth):
            depth = depth.detach().cpu().numpy()
        rgb = np.asarray(rgb)
        depth = np.asarray(depth)
        image_w = rgb.shape[1]
        focal_length = image_w / (2 * np.tan(np.deg2rad(self.fov_deg) / 2))
        center_x = image_w / 2

        boxes = self.detect_bounding_boxes(rgb)
        measurements = []
        for box in boxes:
            u, v = box["center"]
            d_cam = self._sample_depth(depth, u, v)
            if d_cam < 0.0:
                continue
            X, Y = d_cam, (u - center_x) * d_cam / focal_length
            measurements.append([np.hypot(X, Y), np.arctan2(Y, X)])
        return measurements


if __name__ == "__main__":
    import argparse
    from PIL import Image

    parser = argparse.ArgumentParser(description="Run YOLO on an image and print detections as bounding boxes.")
    parser.add_argument("image", nargs="?", default="", help="Optional path to an image file to inspect.")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model checkpoint to load.")
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum detection confidence.")
    args = parser.parse_args()

    detector = YOLOv3TinyPerception(model_path=args.model, conf_thresh=args.conf)

    if args.image and os.path.exists(args.image):
        img = np.array(Image.open(args.image).convert("RGB"))
        boxes = detector.detect_bounding_boxes(img)
        print(boxes)
    else:
        print("No image path supplied. Use: python yolov3_backbone.py path/to/image.jpg")