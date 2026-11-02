"""
Model Factory for Lunar Crater Object Detection.
Instantiates lightweight SSDLite320 with MobileNetV3-Large backbone.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

from ..utils.logging import get_logger

logger = get_logger("model_factory")


def count_parameters(model: nn.Module) -> Tuple[int, int, float]:
    """
    Count total parameters, trainable parameters, and model size in MB.
    Returns: (total_params, trainable_params, size_mb)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 4 bytes per FP32 parameter
    size_mb = (total * 4) / (1024 * 1024)
    return total, trainable, size_mb


def build_crater_detector(
    num_classes: int = 2,
    pretrained_backbone: bool = True,
) -> nn.Module:
    """
    Build SSDLite320 detector with MobileNetV3-Large backbone for crater detection.
    num_classes: 2 (0: background, 1: lunar crater)
    """
    logger.info(f"Building SSDLite320 MobileNetV3-Large (num_classes={num_classes}, pretrained={pretrained_backbone})")

    weights_bb = MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None

    # Torchvision SSDLite320 constructor directly accepts weights_backbone and num_classes
    model = ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone=weights_bb,
        num_classes=num_classes,
    )

    total_p, trainable_p, size_mb = count_parameters(model)
    logger.info(f"Model constructed: {total_p:,} total parameters ({trainable_p:,} trainable), ~{size_mb:.2f} MB in FP32.")
    return model
