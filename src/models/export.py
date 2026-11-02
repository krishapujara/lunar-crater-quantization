"""
ONNX export pipeline for SSDLite320 crater detector.
Uses TorchScript-based tracing compatible with SSD anchor generators and NMS heads.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from ..utils.checksums import compute_sha256
from ..utils.logging import get_logger

logger = get_logger("model_export")


def export_detector_to_onnx(
    model: nn.Module,
    output_onnx_path: Union[str, Path],
    input_resolution: Tuple[int, int] = (320, 320),
    opset_version: int = 14,
    device: str = "cpu",
) -> Tuple[Path, Dict[str, Any]]:
    """
    Export PyTorch SSDLite320 model to ONNX format.
    Validates model structure and executes verification inference in ONNX Runtime.
    """
    out_path = Path(output_onnx_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.to(device)

    # Dummy input representing single batch image [C, H, W]
    dummy_input = [torch.rand(3, input_resolution[0], input_resolution[1], device=device)]

    logger.info(f"Exporting SSDLite320 to ONNX at: {out_path} (opset {opset_version})")

    # Export using legacy TorchScript tracing (dynamo=False) to avoid data-dependent NMS symbol errors
    torch.onnx.export(
        model,
        (dummy_input,),
        str(out_path),
        dynamo=False,
        opset_version=opset_version,
        input_names=["images"],
        output_names=["boxes", "scores", "labels"],
        dynamic_shapes=None,
    )

    # Validate ONNX model with onnx.checker
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)
    logger.info(f"ONNX model structure verified with onnx.checker.")

    # Validate execution with ONNX Runtime
    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    inp_name = session.get_inputs()[0].name
    dummy_np = np.random.rand(3, input_resolution[0], input_resolution[1]).astype(np.float32)

    outputs = session.run(None, {inp_name: dummy_np})
    logger.info(f"ONNX Runtime verification passed! Outputs: {[o.shape for o in outputs]}")

    file_size_bytes = out_path.stat().st_size
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
    sha256_hash = compute_sha256(out_path)

    export_metadata = {
        "export_format": "ONNX",
        "opset_version": opset_version,
        "input_resolution": list(input_resolution),
        "file_size_bytes": file_size_bytes,
        "file_size_mb": file_size_mb,
        "sha256": sha256_hash,
        "input_names": [i.name for i in session.get_inputs()],
        "output_names": [o.name for o in session.get_outputs()],
    }

    # Save metadata JSON
    meta_path = out_path.with_suffix(".json")
    import json
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(export_metadata, f, indent=2)

    return out_path, export_metadata
