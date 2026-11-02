"""
Unified Crater Detection Inference Engine.
Supports PyTorch FP32, ONNX FP32, and Quantized ONNX INT8 models with identical preprocessing.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn

from ..utils.logging import get_logger
from ..data.annotations import scale_boxes

logger = get_logger("inference_engine")


@dataclass
class DetectionResult:
    boxes: np.ndarray  # Shape (N, 4): [xmin, ymin, xmax, ymax] in original image coordinates
    scores: np.ndarray  # Shape (N,): confidence scores in [0.0, 1.0]
    labels: np.ndarray  # Shape (N,): class labels (1: crater)
    diameters_px: np.ndarray  # Shape (N,): estimated pixel diameters
    centers_xy: np.ndarray  # Shape (N, 2): estimated crater centers [cx, cy]
    inference_time_ms: float
    preprocessing_time_ms: float
    postprocessing_time_ms: float
    total_time_ms: float
    model_type: str
    num_detections: int

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Convert detections to a list of dict records for tables or JSON export."""
        records = []
        for i in range(self.num_detections):
            records.append({
                "detection_id": i + 1,
                "confidence": round(float(self.scores[i]), 4),
                "xmin": round(float(self.boxes[i, 0]), 1),
                "ymin": round(float(self.boxes[i, 1]), 1),
                "xmax": round(float(self.boxes[i, 2]), 1),
                "ymax": round(float(self.boxes[i, 3]), 1),
                "center_x": round(float(self.centers_xy[i, 0]), 1),
                "center_y": round(float(self.centers_xy[i, 1]), 1),
                "diameter_px": round(float(self.diameters_px[i]), 1),
                "class_label": "Crater",
            })
        return records


class CraterDetector:
    """
    Unified detector wrapper for lunar crater models.
    Handles PyTorch and ONNX models transparently.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        model_type: Optional[str] = None,
        input_resolution: Tuple[int, int] = (320, 320),
        device: str = "cpu",
    ):
        self.model_path = Path(model_path)
        self.input_resolution = input_resolution
        self.device = device
        self.session: Optional[ort.InferenceSession] = None
        self.torch_model: Optional[nn.Module] = None

        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model artifact not found at: {self.model_path}")

        # Determine type
        suffix = self.model_path.suffix.lower()
        if suffix in [".onnx"]:
            self.model_type = model_type or ("INT8_ONNX" if "int8" in self.model_path.name.lower() else "FP32_ONNX")
            self._load_onnx_session()
        elif suffix in [".pth", ".pt"]:
            self.model_type = "FP32_PyTorch"
            self._load_pytorch_model()
        else:
            raise ValueError(f"Unsupported model format: {suffix}")

        logger.info(f"CraterDetector initialized: {self.model_type} from {self.model_path.name}")

    def _load_onnx_session(self) -> None:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def _load_pytorch_model(self) -> None:
        from .factory import build_crater_detector
        from .checkpoints import load_checkpoint

        self.torch_model = build_crater_detector(num_classes=2, pretrained_backbone=False)
        load_checkpoint(self.model_path, self.torch_model, device=self.device)
        self.torch_model.eval()
        self.torch_model.to(self.device)

    def preprocess(self, image: Union[Image.Image, np.ndarray, Path, str]) -> Tuple[np.ndarray, Tuple[int, int], Image.Image]:
        """Convert input image to standardized (3, 320, 320) float32 array in [0.0, 1.0]."""
        if isinstance(image, (str, Path)):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                pil_img = Image.fromarray(image).convert("RGB")
            else:
                pil_img = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image)}")

        orig_w, orig_h = pil_img.size
        resized = pil_img.resize(self.input_resolution, Image.Resampling.BILINEAR)
        arr = np.array(resized, dtype=np.float32) / 255.0
        # Transpose to (3, H, W)
        tensor_arr = np.transpose(arr, (2, 0, 1))
        return tensor_arr, (orig_w, orig_h), pil_img

    def detect(
        self,
        image: Union[Image.Image, np.ndarray, Path, str],
        confidence_threshold: float = 0.35,
    ) -> DetectionResult:
        """Run crater detection with precise timing breakdown and coordinate rescaling."""
        t0 = time.perf_counter()
        inp_tensor, (orig_w, orig_h), pil_orig = self.preprocess(image)
        t1 = time.perf_counter()

        # Forward inference
        if self.session is not None:
            # ONNX Runtime inference
            outputs = self.session.run(None, {self.input_name: inp_tensor})
            raw_boxes = outputs[0]  # Shape (N, 4)
            raw_scores = outputs[1]  # Shape (N,)
            raw_labels = outputs[2]  # Shape (N,)
        else:
            # PyTorch inference
            with torch.no_grad():
                tensor_torch = torch.from_numpy(inp_tensor).to(self.device)
                preds = self.torch_model([tensor_torch])[0]
                raw_boxes = preds["boxes"].cpu().numpy()
                raw_scores = preds["scores"].cpu().numpy()
                raw_labels = preds["labels"].cpu().numpy()

        t2 = time.perf_counter()

        # Post-processing: confidence thresholding and coordinate scaling
        mask = (raw_scores >= confidence_threshold) & (raw_labels == 1)
        filt_boxes = raw_boxes[mask]
        filt_scores = raw_scores[mask]
        filt_labels = raw_labels[mask]

        if len(filt_boxes) > 0:
            # Rescale boxes from (320, 320) back to original image dimensions (orig_w, orig_h)
            orig_boxes = scale_boxes(filt_boxes, orig_size=self.input_resolution, target_size=(orig_w, orig_h))
            diameters = np.maximum(orig_boxes[:, 2] - orig_boxes[:, 0], orig_boxes[:, 3] - orig_boxes[:, 1])
            centers = np.column_stack([
                (orig_boxes[:, 0] + orig_boxes[:, 2]) / 2.0,
                (orig_boxes[:, 1] + orig_boxes[:, 3]) / 2.0,
            ])
        else:
            orig_boxes = np.zeros((0, 4), dtype=np.float32)
            diameters = np.zeros((0,), dtype=np.float32)
            centers = np.zeros((0, 2), dtype=np.float32)

        t3 = time.perf_counter()

        prep_ms = (t1 - t0) * 1000.0
        infer_ms = (t2 - t1) * 1000.0
        post_ms = (t3 - t2) * 1000.0
        tot_ms = (t3 - t0) * 1000.0

        return DetectionResult(
            boxes=orig_boxes,
            scores=filt_scores,
            labels=filt_labels,
            diameters_px=diameters,
            centers_xy=centers,
            inference_time_ms=infer_ms,
            preprocessing_time_ms=prep_ms,
            postprocessing_time_ms=post_ms,
            total_time_ms=tot_ms,
            model_type=self.model_type,
            num_detections=len(filt_boxes),
        )

    def render_detections(
        self,
        image: Union[Image.Image, np.ndarray, Path, str],
        result: DetectionResult,
        box_color: str = "#00FFCC",
        show_labels: bool = True,
    ) -> Image.Image:
        """Render detection boxes, centers, and confidence badges onto the image."""
        if isinstance(image, (str, Path)):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image).convert("RGB")
        else:
            pil_img = image.convert("RGB").copy()

        draw = ImageDraw.Draw(pil_img)
        w, h = pil_img.size

        # Choose outline width relative to image resolution
        outline_width = max(2, int(min(w, h) / 300))

        for i in range(result.num_detections):
            xmin, ymin, xmax, ymax = result.boxes[i]
            conf = result.scores[i]
            diam = result.diameters_px[i]
            cx, cy = result.centers_xy[i]

            # Draw crater bounding rectangle
            draw.rectangle([xmin, ymin, xmax, ymax], outline=box_color, width=outline_width)

            # Draw center crosshair
            arm = max(3, int(diam * 0.15))
            draw.line([(cx - arm, cy), (cx + arm, cy)], fill="#FFCC00", width=max(1, outline_width - 1))
            draw.line([(cx, cy - arm), (cx, cy + arm)], fill="#FFCC00", width=max(1, outline_width - 1))

            if show_labels:
                label_txt = f"{conf*100:.0f}% ({diam:.0f}px)"
                # Draw small background tag for label readability
                tag_y = max(0, ymin - 16)
                draw.rectangle([xmin, tag_y, xmin + 85, tag_y + 15], fill="#0A1128")
                draw.text((xmin + 3, tag_y + 1), label_txt, fill=box_color)

        return pil_img
