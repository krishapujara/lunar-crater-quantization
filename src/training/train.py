"""
Reproducible Training Pipeline for Lunar Crater Object Detector.
Implements transfer learning, validation evaluation, early stopping, and checkpoint saving.
"""

import math
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import AppConfig
from ..data.dataset import LunarCraterDataset, collate_fn
from ..models.checkpoints import save_checkpoint
from ..models.factory import build_crater_detector
from ..utils.logging import get_logger
from .evaluate import evaluate_model_on_dataset

logger = get_logger("training")


def set_seed(seed: int = 42) -> None:
    """Set random seed across Python, NumPy, and PyTorch for scientific reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_crater_detector(
    cfg: AppConfig,
    train_records: List[Dict[str, Any]],
    val_records: List[Dict[str, Any]],
    progress_callback: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
) -> Tuple[nn.Module, Path, Dict[str, Any]]:
    """
    Execute end-to-end training of SSDLite320 crater detection model.
    Evaluates on validation split each epoch and saves the best model checkpoint.
    """
    set_seed(cfg.training.seed)
    device = torch.device(cfg.training.device if torch.cuda.is_available() and cfg.training.device != "cpu" else "cpu")
    logger.info(f"Starting training on device: {device} for {cfg.training.epochs} epochs")

    # Datasets and Loaders
    train_dataset = LunarCraterDataset(
        records=train_records,
        base_dir=cfg.paths.data_dir,
        target_size=tuple(cfg.model.input_resolution),
        min_diameter_px=cfg.data.min_crater_diameter_px,
    )
    val_dataset = LunarCraterDataset(
        records=val_records,
        base_dir=cfg.paths.data_dir,
        target_size=tuple(cfg.model.input_resolution),
        min_diameter_px=cfg.data.min_crater_diameter_px,
    )

    batch_size = max(1, cfg.training.batch_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Single-process worker for Windows stability
    )

    # Model
    model = build_crater_detector(
        num_classes=cfg.model.num_classes,
        pretrained_backbone=cfg.model.pretrained_backbone,
    )
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.training.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            params,
            lr=cfg.training.learning_rate,
            momentum=cfg.training.momentum,
            weight_decay=cfg.training.weight_decay,
        )

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.epochs,
        eta_min=1e-6,
    )

    best_val_map = -1.0
    best_checkpoint_path = cfg.paths.checkpoints_dir / "crater_detector_baseline_best.pth"
    history: List[Dict[str, Any]] = []

    epochs_without_improvement = 0

    for epoch in range(1, cfg.training.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_losses: List[float] = []
        loss_classifier_list: List[float] = []
        loss_box_reg_list: List[float] = []

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            # In Torchvision SSD models, calling model(images, targets) in train mode returns a dict of losses
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            loss_value = losses.item()
            if not math.isfinite(loss_value):
                logger.warning(f"Loss is {loss_value}, skipping batch.")
                continue

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()

            train_losses.append(loss_value)
            if "classification" in loss_dict:
                loss_classifier_list.append(loss_dict["classification"].item())
            if "bbox_regression" in loss_dict:
                loss_box_reg_list.append(loss_dict["bbox_regression"].item())

        lr_scheduler.step()

        mean_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        mean_cls_loss = float(np.mean(loss_classifier_list)) if loss_classifier_list else 0.0
        mean_reg_loss = float(np.mean(loss_box_reg_list)) if loss_box_reg_list else 0.0

        # Evaluate on validation split
        val_metrics = evaluate_model_on_dataset(
            model,
            val_dataset,
            confidence_threshold=cfg.evaluation.confidence_threshold,
            iou_threshold=cfg.evaluation.iou_threshold,
            device=device,
        )

        val_map = val_metrics.get("ap_50", 0.0)
        epoch_duration = time.perf_counter() - epoch_start

        epoch_stats = {
            "epoch": epoch,
            "train_loss": round(mean_train_loss, 4),
            "cls_loss": round(mean_cls_loss, 4),
            "reg_loss": round(mean_reg_loss, 4),
            "val_ap50": round(val_map, 4),
            "val_map_50_95": round(val_metrics.get("map_50_95", 0.0), 4),
            "val_precision": val_metrics.get("precision", 0.0),
            "val_recall": val_metrics.get("recall", 0.0),
            "learning_rate": round(optimizer.param_groups[0]["lr"], 6),
            "epoch_duration_s": round(epoch_duration, 1),
        }
        history.append(epoch_stats)

        logger.info(
            f"Epoch {epoch}/{cfg.training.epochs} — Train Loss: {mean_train_loss:.4f} "
            f"(Cls: {mean_cls_loss:.4f}, Reg: {mean_reg_loss:.4f}) | "
            f"Val mAP@0.50: {val_map:.4f} | Time: {epoch_duration:.1f}s"
        )

        if progress_callback is not None:
            progress_callback(epoch, cfg.training.epochs, epoch_stats)

        # Check for best model
        if val_map >= best_val_map:
            best_val_map = val_map
            epochs_without_improvement = 0
            save_checkpoint(
                model=model,
                checkpoint_path=best_checkpoint_path,
                optimizer=optimizer,
                epoch=epoch,
                metrics=epoch_stats,
                extra_metadata={
                    "architecture": cfg.model.architecture,
                    "num_classes": cfg.model.num_classes,
                    "best_metric": "val_ap50",
                    "best_value": val_map,
                },
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.training.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epochs_without_improvement} epochs without improvement.")
                break

    # Also save final checkpoint
    final_checkpoint_path = cfg.paths.checkpoints_dir / "crater_detector_baseline_final.pth"
    save_checkpoint(
        model=model,
        checkpoint_path=final_checkpoint_path,
        optimizer=optimizer,
        epoch=len(history),
        metrics=history[-1] if history else {},
    )

    training_summary = {
        "total_epochs_trained": len(history),
        "best_val_ap50": best_val_map,
        "best_checkpoint": str(best_checkpoint_path),
        "final_checkpoint": str(final_checkpoint_path),
        "history": history,
    }

    return model, best_checkpoint_path, training_summary
