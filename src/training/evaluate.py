"""
Comprehensive Evaluation Pipeline.
Evaluates PyTorch and ONNX models on identical held-out test splits.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models.inference import CraterDetector
from ..utils.logging import get_logger
from .metrics import compute_map_range, evaluate_detection_metrics

logger = get_logger("evaluation")


def evaluate_model_on_dataset(
    detector_or_model: Union[CraterDetector, torch.nn.Module],
    dataset: torch.utils.data.Dataset,
    confidence_threshold: float = 0.35,
    iou_threshold: float = 0.50,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Run evaluation across an entire dataset.
    Accepts either a CraterDetector or a PyTorch detection model.
    """
    all_pred_boxes: List[np.ndarray] = []
    all_pred_scores: List[np.ndarray] = []
    all_gt_boxes: List[np.ndarray] = []
    latencies_ms: List[float] = []

    is_crater_detector = isinstance(detector_or_model, CraterDetector)

    if not is_crater_detector:
        detector_or_model.eval()
        detector_or_model.to(device)

    for i in range(len(dataset)):
        img_tensor, target = dataset[i]
        gt_boxes = target["boxes"].cpu().numpy()

        if is_crater_detector:
            # Reconstruct PIL image or pass array
            # img_tensor is (3, H, W) in [0.0, 1.0]
            arr = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            det_res = detector_or_model.detect(arr, confidence_threshold=confidence_threshold)
            pred_b = det_res.boxes
            pred_s = det_res.scores
            latencies_ms.append(det_res.inference_time_ms)
        else:
            t0 = time.perf_counter()
            with torch.no_grad():
                inp = [img_tensor.to(device)]
                preds = detector_or_model(inp)[0]
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

            raw_b = preds["boxes"].cpu().numpy()
            raw_s = preds["scores"].cpu().numpy()
            raw_l = preds["labels"].cpu().numpy()

            mask = (raw_s >= confidence_threshold) & (raw_l == 1)
            pred_b = raw_b[mask]
            pred_s = raw_s[mask]

        all_pred_boxes.append(pred_b)
        all_pred_scores.append(pred_s)
        all_gt_boxes.append(gt_boxes)

    # Compute metrics at IoU 0.50
    metrics = evaluate_detection_metrics(
        all_pred_boxes,
        all_pred_scores,
        all_gt_boxes,
        iou_threshold=iou_threshold,
    )

    # Compute mAP@0.50:0.95
    map_50_95 = compute_map_range(
        all_pred_boxes,
        all_pred_scores,
        all_gt_boxes,
        start_iou=0.50,
        end_iou=0.95,
        step_iou=0.05,
    )
    metrics["map_50_95"] = map_50_95

    metrics["mean_latency_ms"] = round(float(np.mean(latencies_ms)), 2)
    metrics["median_latency_ms"] = round(float(np.median(latencies_ms)), 2)
    metrics["evaluated_samples"] = len(dataset)
    metrics["confidence_threshold"] = confidence_threshold
    metrics["iou_threshold"] = iou_threshold

    logger.info(
        f"Evaluation finished: mAP@0.50={metrics['ap_50']}, mAP@0.50:0.95={metrics['map_50_95']}, "
        f"P={metrics['precision']}, R={metrics['recall']}, Mean Latency={metrics['mean_latency_ms']} ms."
    )
    return metrics
