"""
Scientific Object Detection Metrics for Lunar Crater Detection.
Calculates IoU, Precision, Recall, F1, AP@0.50, and mAP@0.50:0.95.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Intersection over Union (IoU) between two sets of boxes [xmin, ymin, xmax, ymax].
    boxes_a: Shape (N, 4)
    boxes_b: Shape (M, 4)
    Returns: Shape (N, M)
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    # Intersection coordinates
    xmin = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    ymin = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    xmax = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    ymax = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter_w = np.maximum(0.0, xmax - xmin)
    inter_h = np.maximum(0.0, ymax - ymin)
    intersection = inter_w * inter_h

    # Areas
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union = area_a[:, None] + area_b[None, :] - intersection
    union = np.maximum(union, 1e-8)

    return intersection / union


def match_detections(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float = 0.50,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Match predictions to ground truth boxes using greedy IoU assignment.
    Returns:
    - tp_flags: boolean array of shape (N_pred,) indicating true positives
    - fp_flags: boolean array of shape (N_pred,) indicating false positives
    - n_gt: total number of ground truth craters
    """
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)

    if n_pred == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=bool), n_gt

    if n_gt == 0:
        return np.zeros(n_pred, dtype=bool), np.ones(n_pred, dtype=bool), 0

    # Sort predictions by descending confidence
    sort_idx = np.argsort(-pred_scores)
    sorted_boxes = pred_boxes[sort_idx]

    iou_matrix = compute_iou_matrix(sorted_boxes, gt_boxes)

    tp = np.zeros(n_pred, dtype=bool)
    fp = np.zeros(n_pred, dtype=bool)
    gt_matched = np.zeros(n_gt, dtype=bool)

    for i in range(n_pred):
        best_gt_idx = -1
        best_iou = -1.0

        for j in range(n_gt):
            if not gt_matched[j] and iou_matrix[i, j] >= iou_threshold:
                if iou_matrix[i, j] > best_iou:
                    best_iou = iou_matrix[i, j]
                    best_gt_idx = j

        if best_gt_idx >= 0:
            tp[i] = True
            gt_matched[best_gt_idx] = True
        else:
            fp[i] = True

    # Invert sorting to match original order
    inv_sort_idx = np.argsort(sort_idx)
    return tp[inv_sort_idx], fp[inv_sort_idx], n_gt


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute Average Precision using the continuous area under the precision-recall curve."""
    # Append sentinel values at ends
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))

    # Compute maximum precision envelope
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Find points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def evaluate_detection_metrics(
    all_pred_boxes: List[np.ndarray],
    all_pred_scores: List[np.ndarray],
    all_gt_boxes: List[np.ndarray],
    iou_threshold: float = 0.50,
) -> Dict[str, float]:
    """
    Calculate detection metrics across an entire dataset split:
    Precision, Recall, F1, and AP@IoU.
    """
    total_gt = 0
    flat_scores: List[float] = []
    flat_tp: List[bool] = []
    flat_fp: List[bool] = []

    for pred_b, pred_s, gt_b in zip(all_pred_boxes, all_pred_scores, all_gt_boxes):
        tp, fp, n_gt = match_detections(pred_b, pred_s, gt_b, iou_threshold=iou_threshold)
        total_gt += n_gt
        flat_scores.extend(pred_s.tolist())
        flat_tp.extend(tp.tolist())
        flat_fp.extend(fp.tolist())

    if len(flat_scores) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "ap_50": 0.0,
            "total_predictions": 0,
            "total_ground_truth": total_gt,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": total_gt,
        }

    scores_arr = np.array(flat_scores)
    tp_arr = np.array(flat_tp, dtype=bool)
    fp_arr = np.array(flat_fp, dtype=bool)

    # Sort all predictions globally by descending confidence
    sort_idx = np.argsort(-scores_arr)
    tp_sorted = tp_arr[sort_idx]
    fp_sorted = fp_arr[sort_idx]

    cum_tp = np.cumsum(tp_sorted)
    cum_fp = np.cumsum(fp_sorted)

    recalls = cum_tp / max(total_gt, 1)
    precisions = cum_tp / (cum_tp + cum_fp)

    ap = compute_ap(recalls, precisions)

    total_tp = int(cum_tp[-1]) if len(cum_tp) > 0 else 0
    total_fp = int(cum_fp[-1]) if len(cum_fp) > 0 else 0
    total_fn = max(0, total_gt - total_tp)

    final_precision = float(total_tp / max(1, total_tp + total_fp))
    final_recall = float(total_tp / max(1, total_gt))
    final_f1 = (
        float(2 * final_precision * final_recall / max(1e-8, final_precision + final_recall))
        if (final_precision + final_recall) > 0
        else 0.0
    )

    return {
        "precision": round(final_precision, 4),
        "recall": round(final_recall, 4),
        "f1_score": round(final_f1, 4),
        "ap_50": round(ap, 4),
        "total_predictions": len(flat_scores),
        "total_ground_truth": total_gt,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
    }


def compute_map_range(
    all_pred_boxes: List[np.ndarray],
    all_pred_scores: List[np.ndarray],
    all_gt_boxes: List[np.ndarray],
    start_iou: float = 0.50,
    end_iou: float = 0.95,
    step_iou: float = 0.05,
) -> float:
    """Compute mAP averaged across IoU thresholds from start_iou to end_iou (mAP@0.50:0.95)."""
    ious = np.arange(start_iou, end_iou + 1e-5, step_iou)
    aps = []
    for thresh in ious:
        metrics = evaluate_detection_metrics(all_pred_boxes, all_pred_scores, all_gt_boxes, iou_threshold=float(thresh))
        aps.append(metrics["ap_50"])
    return round(float(np.mean(aps)), 4)
