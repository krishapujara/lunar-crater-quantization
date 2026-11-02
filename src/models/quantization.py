"""
Post-Training Quantization (PTQ) pipeline for SSDLite320 Lunar Crater Detector.
Converts FP32 weights to INT8, inspects operator support, and verifies ONNX Runtime inference.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

from ..utils.checksums import compute_sha256
from ..utils.logging import get_logger

logger = get_logger("quantization")


def inspect_onnx_operators(onnx_path: Union[str, Path]) -> Dict[str, int]:
    """Inspect and count all operators present in an ONNX model graph."""
    model = onnx.load(str(onnx_path))
    counts: Dict[str, int] = {}
    for node in model.graph.node:
        op = node.op_type
        counts[op] = counts.get(op, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))


def quantize_fp32_to_int8(
    input_fp32_onnx: Union[str, Path],
    output_int8_onnx: Union[str, Path],
    weight_type: str = "QInt8",
) -> Tuple[Path, Dict[str, Any]]:
    """
    Apply Post-Training Dynamic Quantization (PTQ) from FP32 to INT8.
    Quantizes computationally heavy Conv, Gemm, and MatMul weights to INT8.
    Verifies exported artifact by executing actual inference in ONNX Runtime.
    """
    fp32_path = Path(input_fp32_onnx)
    int8_path = Path(output_int8_onnx)
    int8_path.parent.mkdir(parents=True, exist_ok=True)

    if not fp32_path.is_file():
        raise FileNotFoundError(f"Input FP32 ONNX model not found: {fp32_path}")

    logger.info(f"Initiating INT8 quantization: {fp32_path} -> {int8_path}")

    # Inspect initial operators
    initial_ops = inspect_onnx_operators(fp32_path)
    fp32_size_bytes = fp32_path.stat().st_size
    fp32_size_mb = round(fp32_size_bytes / (1024 * 1024), 2)

    # Perform post-training dynamic quantization
    q_type = QuantType.QInt8 if weight_type == "QInt8" else QuantType.QUInt8
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=q_type,
    )

    if not int8_path.is_file():
        raise RuntimeError("Quantization failed: Output INT8 artifact was not created.")

    int8_size_bytes = int8_path.stat().st_size
    int8_size_mb = round(int8_size_bytes / (1024 * 1024), 2)
    abs_reduction_bytes = fp32_size_bytes - int8_size_bytes
    pct_reduction = round(((fp32_size_bytes - int8_size_bytes) / fp32_size_bytes) * 100.0, 2)

    # Inspect quantized operators
    quantized_ops = inspect_onnx_operators(int8_path)

    # Operator support classification
    quantized_op_names = [op for op in quantized_ops if "Integer" in op or "Quantize" in op]
    # In dynamic quantization, Conv/Gemm nodes use INT8 weights while coordinate and non-linear transforms remain in float/int64
    precision_mode = "Mixed-Precision (INT8 Weights / FP32 Activations & NMS)"

    # Verify execution with ONNX Runtime
    session = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    inp_name = session.get_inputs()[0].name
    dummy_np = np.random.rand(3, 320, 320).astype(np.float32)

    try:
        out = session.run(None, {inp_name: dummy_np})
        inference_verified = True
        sample_output_shapes = [list(o.shape) if hasattr(o, "shape") else len(o) for o in out]
    except Exception as e:
        logger.error(f"Inference execution failed on quantized model: {e}")
        inference_verified = False
        sample_output_shapes = []

    report = {
        "framework": "ONNX Runtime",
        "quantization_version": ort.__version__,
        "quantization_method": "Post-Training Dynamic Quantization (PTQ)",
        "weight_type": weight_type,
        "precision_mode": precision_mode,
        "fp32_model_size_bytes": fp32_size_bytes,
        "fp32_model_size_mb": fp32_size_mb,
        "int8_model_size_bytes": int8_size_bytes,
        "int8_model_size_mb": int8_size_mb,
        "absolute_reduction_bytes": abs_reduction_bytes,
        "percentage_size_reduction": pct_reduction,
        "fp32_sha256": compute_sha256(fp32_path),
        "int8_sha256": compute_sha256(int8_path),
        "inference_verified": inference_verified,
        "sample_output_shapes": sample_output_shapes,
        "initial_operators": initial_ops,
        "quantized_operators": quantized_ops,
        "unquantized_operators_note": (
            "Operators such as Slice, NonZero, Greater, and NMS bounding box clipping "
            "operate on coordinate index tensors and remain unquantized by design to preserve spatial precision."
        ),
    }

    # Save metadata JSON
    meta_path = int8_path.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(
        f"Quantization successful: {fp32_size_mb} MB -> {int8_size_mb} MB ({pct_reduction}% reduction). "
        f"Inference verified: {inference_verified}."
    )
    return int8_path, report
