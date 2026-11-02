"""
CLI Script: Export PyTorch SSDLite320 to ONNX and Post-Train Quantize to INT8.
Usage: python scripts/quantize_model.py [--checkpoint path/to/model.pth]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.models.checkpoints import load_checkpoint
from src.models.export import export_detector_to_onnx
from src.models.factory import build_crater_detector
from src.models.quantization import quantize_fp32_to_int8
from src.utils.logging import get_logger

logger = get_logger("cli_quantize")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and Quantize Crater Detector")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to PyTorch checkpoint")
    args = parser.parse_args()

    cfg = load_config()
    ckpt_path = Path(args.checkpoint) if args.checkpoint else cfg.paths.checkpoints_dir / "crater_detector_baseline_best.pth"

    if not ckpt_path.is_file():
        logger.error(
            f"Checkpoint not found at {ckpt_path}. "
            "Please run 'python scripts/train_model.py' first or provide a valid --checkpoint."
        )
        sys.exit(1)

    logger.info("Step 1: Loading PyTorch model checkpoint...")
    model = build_crater_detector(num_classes=cfg.model.num_classes, pretrained_backbone=cfg.model.pretrained_backbone)
    load_checkpoint(ckpt_path, model, device="cpu")

    logger.info("Step 2: Exporting to FP32 ONNX format...")
    fp32_onnx_path = cfg.paths.quantized_dir / "crater_detector_fp32.onnx"
    export_detector_to_onnx(
        model=model,
        output_onnx_path=fp32_onnx_path,
        input_resolution=tuple(cfg.model.input_resolution),
        opset_version=cfg.quantization.opset_version,
    )

    logger.info("Step 3: Quantizing to INT8 using ONNX Runtime Post-Training Dynamic Quantization...")
    int8_onnx_path = cfg.paths.quantized_dir / "crater_detector_int8.onnx"
    out_path, report = quantize_fp32_to_int8(
        input_fp32_onnx=fp32_onnx_path,
        output_int8_onnx=int8_onnx_path,
        weight_type=cfg.quantization.weight_type,
    )

    print(f"\n--- Quantization Summary ---")
    print(f"FP32 Model Size: {report['fp32_model_size_mb']:.2f} MB")
    print(f"INT8 Model Size: {report['int8_model_size_mb']:.2f} MB")
    print(f"Storage Reduction: -{report['percentage_size_reduction']:.1f}%")
    print(f"Inference Verified: {report['inference_verified']}")
    print(f"Exported INT8 Model: {out_path}")


if __name__ == "__main__":
    main()
