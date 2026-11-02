"""
Model checkpoint management, serialization, and cryptographic integrity verification.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from ..utils.checksums import compute_sha256
from ..utils.logging import get_logger

logger = get_logger("model_checkpoints")


def save_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save PyTorch model weights along with training metadata and SHA-256 hash."""
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics or {},
        "metadata": extra_metadata or {},
        "torch_version": torch.__version__,
    }

    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(payload, path)
    sha256_hash = compute_sha256(path)

    # Save adjacent metadata JSON for lightweight inspection without loading full torch weights
    meta_path = path.with_suffix(".json")
    meta_info = {
        "checkpoint_file": path.name,
        "epoch": epoch,
        "metrics": metrics or {},
        "metadata": extra_metadata or {},
        "sha256": sha256_hash,
        "file_size_bytes": path.stat().st_size,
        "file_size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        "torch_version": torch.__version__,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)

    logger.info(f"Checkpoint saved: {path} (SHA-256: {sha256_hash[:12]}..., Size: {meta_info['file_size_mb']} MB)")
    return path


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Safely load model weights into model instance, handling CPU/GPU mapping."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    logger.info(f"Loading checkpoint from: {path} onto {device}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Direct state dict
        model.load_state_dict(checkpoint)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def get_checkpoint_info(checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
    """Read metadata JSON if available, or compute on the fly."""
    path = Path(checkpoint_path)
    meta_path = path.with_suffix(".json")

    if meta_path.is_file():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if path.is_file():
        size_bytes = path.stat().st_size
        return {
            "checkpoint_file": path.name,
            "sha256": compute_sha256(path),
            "file_size_bytes": size_bytes,
            "file_size_mb": round(size_bytes / (1024 * 1024), 2),
        }

    return {"error": "Checkpoint file not found"}
