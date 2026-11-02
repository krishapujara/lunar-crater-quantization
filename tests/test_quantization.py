"""
Tests for ONNX export, INT8 post-training quantization, and ONNX Runtime execution.
"""

from pathlib import Path
import pytest
import torch

from src.models.export import export_detector_to_onnx
from src.models.factory import build_crater_detector
from src.models.inference import CraterDetector
from src.models.quantization import inspect_onnx_operators, quantize_fp32_to_int8


@pytest.fixture(scope="module")
def exported_models(tmp_path_factory):
    """Module-level fixture exporting FP32 ONNX and generating INT8 quantized model."""
    tmp = tmp_path_factory.mktemp("quant_test")
    model = build_crater_detector(num_classes=2, pretrained_backbone=False)

    fp32_onnx = tmp / "model_fp32.onnx"
    int8_onnx = tmp / "model_int8.onnx"

    # Export FP32 ONNX
    export_detector_to_onnx(model, fp32_onnx, input_resolution=(320, 320), opset_version=14)

    # Quantize to INT8
    _, report = quantize_fp32_to_int8(fp32_onnx, int8_onnx, weight_type="QInt8")

    return fp32_onnx, int8_onnx, report


def test_onnx_export_structure(exported_models):
    """Test that FP32 ONNX model exists and has valid operators."""
    fp32_onnx, _, _ = exported_models
    assert fp32_onnx.is_file()

    ops = inspect_onnx_operators(fp32_onnx)
    assert "Conv" in ops
    assert "Relu" in ops or "Clip" in ops or "HardSigmoid" in ops


def test_int8_quantization_reduction_and_verification(exported_models):
    """Test INT8 model file size reduction and verification report."""
    fp32_onnx, int8_onnx, report = exported_models
    assert int8_onnx.is_file()

    assert report["percentage_size_reduction"] > 50.0  # Typically ~70% reduction
    assert report["int8_model_size_mb"] < report["fp32_model_size_mb"]
    assert report["inference_verified"] is True
    assert "Mixed-Precision" in report["precision_mode"]


def test_onnx_detector_inference(exported_models):
    """Test CraterDetector running on both exported ONNX models."""
    fp32_onnx, int8_onnx, _ = exported_models

    fp32_det = CraterDetector(fp32_onnx)
    int8_det = CraterDetector(int8_onnx)

    assert fp32_det.model_type == "FP32_ONNX"
    assert int8_det.model_type == "INT8_ONNX"

    # Run inference on dummy 320x320 image
    import numpy as np
    dummy_img = np.full((320, 320, 3), 100, dtype=np.uint8)

    res_fp32 = fp32_det.detect(dummy_img, confidence_threshold=0.01)
    res_int8 = int8_det.detect(dummy_img, confidence_threshold=0.01)

    assert res_fp32.inference_time_ms > 0
    assert res_int8.inference_time_ms > 0
