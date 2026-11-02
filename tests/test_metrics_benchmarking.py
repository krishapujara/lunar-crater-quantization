"""
Tests for object detection metrics (IoU, AP@50, mAP) and benchmarking suites.
"""

from pathlib import Path
import numpy as np
import pytest

from src.benchmarking.benchmark import (
    BenchmarkMetrics,
    ComparisonReport,
    benchmark_single_model,
    get_system_hardware_info,
)
from src.benchmarking.reporting import save_benchmark_report
from src.training.metrics import (
    compute_ap,
    compute_iou_matrix,
    compute_map_range,
    evaluate_detection_metrics,
    match_detections,
)


def test_iou_matrix_known_fixtures():
    """Test pairwise IoU computation with known bounding boxes."""
    # Box A: [0, 0, 10, 10], Area = 100
    # Box B: [0, 0, 10, 10], Identical -> IoU = 1.0
    # Box C: [5, 0, 15, 10], Overlaps [5..10, 0..10] -> Inter = 50, Union = 150 -> IoU = 1/3
    # Box D: [20, 20, 30, 30], Disjoint -> IoU = 0.0

    boxes_a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    boxes_b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=np.float32)

    iou_mat = compute_iou_matrix(boxes_a, boxes_b)
    assert iou_mat.shape == (1, 3)
    np.testing.assert_allclose(iou_mat[0, 0], 1.0, atol=1e-5)
    np.testing.assert_allclose(iou_mat[0, 1], 1.0 / 3.0, atol=1e-5)
    np.testing.assert_allclose(iou_mat[0, 2], 0.0, atol=1e-5)


def test_match_detections_tp_fp():
    """Test matching predictions against ground-truth boxes."""
    pred_boxes = np.array([[0, 0, 10, 10], [100, 100, 120, 120]], dtype=np.float32)
    pred_scores = np.array([0.9, 0.8], dtype=np.float32)
    gt_boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)

    tp, fp, n_gt = match_detections(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5)
    assert n_gt == 1
    assert tp[0] is np.True_ or tp[0] == 1  # Matches gt
    assert fp[1] is np.True_ or fp[1] == 1  # False positive


def test_compute_ap():
    """Test Average Precision calculation."""
    # Perfect precision and recall
    recalls = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    precisions = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    ap = compute_ap(recalls, precisions)
    assert np.isclose(ap, 1.0)


def test_evaluate_detection_metrics_empty():
    """Test metric evaluation with empty predictions."""
    res = evaluate_detection_metrics(
        all_pred_boxes=[np.zeros((0, 4))],
        all_pred_scores=[np.zeros(0)],
        all_gt_boxes=[np.array([[0, 0, 10, 10]])],
    )
    assert res["precision"] == 0.0
    assert res["recall"] == 0.0
    assert res["false_negatives"] == 1


def test_hardware_info_collector():
    """Test system specification extraction."""
    info = get_system_hardware_info()
    assert "platform" in info
    assert info["cpu_count_logical"] >= 1
    assert info["ram_total_gb"] > 0


def test_save_benchmark_report(tmp_path: Path):
    """Test saving benchmark comparison report as JSON and CSV."""
    m_fp32 = BenchmarkMetrics(
        model_name="model_fp32.onnx",
        model_type="FP32_ONNX",
        runtime="ONNX Runtime",
        device="cpu",
        repetition_count=10,
        warmup_count=2,
        file_size_mb=14.27,
        mean_latency_ms=45.2,
        median_latency_ms=44.1,
        std_latency_ms=2.1,
        p90_latency_ms=47.0,
        p95_latency_ms=48.2,
        min_latency_ms=41.0,
        max_latency_ms=51.0,
        mean_prep_ms=1.2,
        mean_post_ms=0.8,
        throughput_fps=22.1,
        process_memory_mb=180.5,
        latencies=[44.0, 45.0],
    )
    m_int8 = BenchmarkMetrics(
        model_name="model_int8.onnx",
        model_type="INT8_ONNX",
        runtime="ONNX Runtime",
        device="cpu",
        repetition_count=10,
        warmup_count=2,
        file_size_mb=4.08,
        mean_latency_ms=32.1,
        median_latency_ms=31.5,
        std_latency_ms=1.5,
        p90_latency_ms=33.5,
        p95_latency_ms=34.2,
        min_latency_ms=30.0,
        max_latency_ms=36.0,
        mean_prep_ms=1.2,
        mean_post_ms=0.8,
        throughput_fps=31.2,
        process_memory_mb=182.0,
        latencies=[31.0, 32.0],
    )

    report = ComparisonReport(
        fp32_metrics=m_fp32,
        int8_metrics=m_int8,
        size_reduction_mb=10.19,
        size_reduction_pct=71.4,
        latency_reduction_pct=28.5,
        speedup_ratio=1.40,
        system_info=get_system_hardware_info(),
    )

    json_p, csv_p = save_benchmark_report(report, tmp_path, report_name="test_report")
    assert json_p.is_file()
    assert csv_p.is_file()
