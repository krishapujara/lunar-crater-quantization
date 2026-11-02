"""
Tests for model factory, parameter counting, checkpoint handling, and inference engine.
"""

from pathlib import Path
import numpy as np
import pytest
import torch

from src.models.checkpoints import get_checkpoint_info, load_checkpoint, save_checkpoint
from src.models.factory import build_crater_detector, count_parameters
from src.models.inference import CraterDetector, DetectionResult


def test_model_factory_and_parameter_count():
    """Test instantiating SSDLite320 model and verifying parameter counts."""
    model = build_crater_detector(num_classes=2, pretrained_backbone=False)
    assert isinstance(model, torch.nn.Module)

    total_p, trainable_p, size_mb = count_parameters(model)
    assert total_p > 1_000_000  # SSDLite320 with MobileNetV3 is ~3-4M params
    assert trainable_p == total_p
    assert 5.0 < size_mb < 30.0


def test_checkpoint_saving_and_loading(tmp_path: Path):
    """Test saving checkpoint with metadata and loading back safely."""
    model = build_crater_detector(num_classes=2, pretrained_backbone=False)
    ckpt_path = tmp_path / "test_model.pth"

    metrics = {"val_ap50": 0.582, "train_loss": 0.312}
    save_checkpoint(model, ckpt_path, epoch=1, metrics=metrics)

    assert ckpt_path.is_file()
    meta_path = ckpt_path.with_suffix(".json")
    assert meta_path.is_file()

    info = get_checkpoint_info(ckpt_path)
    assert info["metrics"]["val_ap50"] == 0.582
    assert "sha256" in info

    # Test loading
    new_model = build_crater_detector(num_classes=2, pretrained_backbone=False)
    loaded = load_checkpoint(ckpt_path, new_model, device="cpu")
    assert loaded["epoch"] == 1


def test_crater_detector_inference_pipeline(tmp_path: Path):
    """Test CraterDetector preprocessing, prediction, coordinate rescaling, and rendering."""
    model = build_crater_detector(num_classes=2, pretrained_backbone=False)
    ckpt_path = tmp_path / "sample_model.pth"
    save_checkpoint(model, ckpt_path, epoch=1)

    detector = CraterDetector(ckpt_path, device="cpu")
    assert detector.model_type == "FP32_PyTorch"

    # Synthetic image (400x500 pixels)
    img = np.full((500, 400, 3), 120, dtype=np.uint8)
    res = detector.detect(img, confidence_threshold=0.01)

    assert isinstance(res, DetectionResult)
    assert res.inference_time_ms > 0
    assert res.total_time_ms >= res.inference_time_ms
    assert len(res.boxes.shape) == 2
    assert res.boxes.shape[1] == 4 if len(res.boxes) > 0 else True

    # Test rendering
    rendered = detector.render_detections(img, res)
    assert rendered.size == (400, 500)
